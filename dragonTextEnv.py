import pandas as pd

from gym_dragon.envs import MiniDragonEnv
from gym_dragon.core import Region, Agent, Tool, Bomb
from gym_dragon.wrappers import MiniObs
import openai
import time
import json
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

COLOR_TO_STR={0: 'Red',1:'Green',2:'Blue'}
# ACTION_TO_STR={1: 'inspect_bomb',7:'go_to_node_0',8:'go_to_node_3',9:'go_to_node_5',10:'go_to_node_6',11:'go_to_node_8'}
BOMBSTATE_TO_STR = {0: 'inactive',1: 'active',2: 'exploded',3: 'defused'}
openai.api_key = os.environ.get("OPENAI_API_KEY", "na")
class DragonTextEnv():
    def __init__(self,seed = None, include_agent_action = False,allow_comm = True,act_and_comm = True,tool_per_agent = 2):
        self.seed = seed

        self.valid_node = [0,3,5,6,8]
        self.include_agent_action = include_agent_action
        self.allow_comm = allow_comm
        self.act_and_comm = act_and_comm
        self.tool_per_agent = tool_per_agent

        self.env = MiniDragonEnv(mission_length = 999,
                        recon_phase_length=0,
                         include_chained_bombs=False,
                         include_fire_bombs=False,
                         include_fuse_bombs=False,
                         color_tools_only=True,
                        obs_wrapper=MiniObs)
        self.env.seed(self.seed)

        if self.tool_per_agent == 2:
            self.env.reset(csv_path =None,
                           num_bombs_per_region = 5,
                           start_location = None,
                           start_regions=set(Region.village),
                           tool_allocation = {'alpha':{Tool.red:99,Tool.green:99},'bravo':{Tool.blue:99,Tool.green:99},'charlie':{Tool.red:99,Tool.blue:99}})
        else:
            self.env.reset(csv_path =None,
                           num_bombs_per_region = 5,
                           start_location = None,
                           start_regions=set(Region.village),
                           tool_allocation = {'alpha':{Tool.red:99},'bravo':{Tool.green:99},'charlie':{Tool.blue:99}})


    def step(self,agent_id, round, initial_actions, communications):
        # action is object
        # agent is str index
        reward = {agent_id: 0 for agent_id in self.env.agents.keys()}
        info = {agent_id: {} for agent_id in self.env.agents.keys()}
        prev_agent_health = {agent.id: agent.health for agent in self.env.agents.values()}

        action = initial_actions[agent_id]
        obs_text = 'Your action is invalid.'

        Action = self.env.action_enum
        agent = self.env.agents[agent_id]
        if action is None:
            obs_text = "Invalid action."
        elif action == Action.inspect_bomb:
            if agent.bomb:
                bomb_id = str(agent.bomb.id)
                sequence = '-'.join([COLOR_TO_STR[x] for x in agent.bomb._full_sequence[agent.bomb._current_step:]])
                num_stage = str(agent.bomb.num_stages)
                obs_text = "You inspected Bomb {bomb_id}. This bomb is a {num_stage}-stage bomb and its remaining sequence is {sequence}.".format(
                    bomb_id=bomb_id, num_stage=num_stage, sequence=sequence)

                agent.bomb.inspect()
                self.env._inspected_bombs[agent.id].add(agent.bomb)
                self.env.observations[agent.id].update_from_inspection()
            else:
                current_room = str(agent.node.id)
                obs_text = "There is no bomb in the current current location, Room {current_room}, for you to inspect.".format(
                    current_room=current_room)
        elif action.node() is not None:
            # Go to the given node
            if self.env._get_action_mask(agent_id)[action] or action.node().id == agent.node.id:
                room_id = str(action.node().id)
                room_agents = ' and '.join([str(x.id) for x in action.node().agents])
                room_bombs = ' and '.join(['Bomb ' + str(x.id) for x in action.node().bombs])
                obs_text = "You moved to Room {room_id}. In the new room you found {room_agents}, {room_bombs}.".format(
                    room_id=room_id, room_agents=room_agents, room_bombs=room_bombs)

                agent.go_to(action.node())
            else:
                room_id = str(action.node().id)
                current_room = str(agent.node.id)
                obs_text = "You can not directly move to Room {room_id} because it is not adjacent to your current location, Room {current_room}. Consider taking a detour to another room first and then move to your destination.".format(
                    room_id=room_id, current_room=current_room)


        elif action.tool() is not None:
            tool = agent.get_tool_from_inventory(action.tool())
            if tool in Tool.bomb_tools():
                if agent.bomb:
                    if tool.color == agent.bomb.color:
                        agent.bomb.apply_tool(tool)
                        tool_color = COLOR_TO_STR[tool.color]
                        bomb_id = str(agent.bomb.id)
                        sequence = '-'.join(
                            [COLOR_TO_STR[x] for x in agent.bomb._full_sequence[agent.bomb._current_step:]])
                        state = BOMBSTATE_TO_STR[agent.bomb.state]

                        if agent.bomb.state == Bomb.BombState.defused:
                            reward = agent.bomb.value
                            obs_text = "You applied Tool {tool_color} to Bomb {bomb_id}. This bomb is defused successfully.".format(
                                tool_color=tool_color, bomb_id=bomb_id)
                        else:
                            obs_text = "You applied Tool {tool_color} to Bomb {bomb_id}. This bomb is {state} and its remaining sequence is {sequence}.".format(
                                tool_color=tool_color, bomb_id=bomb_id, state=state, sequence=sequence)

                    else:
                        tool_color = COLOR_TO_STR[tool.color]
                        bomb_id = agent.bomb.id
                        sequence = agent.bomb._full_sequence[agent.bomb._current_step:]
                        state = agent.bomb.state

                        obs_text = "You can not apply Tool {tool_color} to Bomb {bomb_id} because the sequence of this bomb is {sequence}. You will need to apply other color tool first.".format(
                            tool_color=tool_color, bomb_id=bomb_id, sequence=sequence)
                else:
                    obs_text = "There is no bomb in your current location, room {room_id}, for you to defuse.".format(
                        room_id=agent.node.id)
            else:
                obs_text = "You do not have {tool}. Consider asking your teammates who have this tool to help you defuse the bomb.".format(
                    tool=COLOR_TO_STR[action.tool()])


        self.env.tick()


        for agent_id in info:
            info[agent_id]['score'] = self.env.score

        obs, reward, done, info = self.env._get_obs(), reward, self.env._get_done(), info

        # team_location_text = 'Player alpha is in Room {loc_a}; Player bravo is in Room {loc_b}; Player charlie is in Room {loc_c}.'.format(
        #     loc_a=str(self.env.agents['alpha'].node.id), loc_b=str(self.env.agents['bravo'].node.id),
        #     loc_c=str(self.env.agents['charlie'].node.id))

        room_id = str(agent.node.id)
        room_agents = ' and '.join([str(x.id) for x in agent.node.agents])
        room_bombs = ' and '.join(['Bomb ' + str(x.id) for x in agent.node.bombs])
        room_contents = "You are currently in Room {room_id}. Contents of this room include {room_agents}, {room_bombs}.".format(
            room_id=room_id, room_agents=room_agents, room_bombs=room_bombs)

        text = 'Your observation is: '
        text += 'Round: {timestep}. '.format(timestep=str(round))
        text += 'Total team score: {score}. '.format(score=str(self.env.score))
        text += 'Results: ' + obs_text + ' '
        text += 'Room contents: '+ room_contents+ ' '
        # text += 'Teammate Locations: ' + team_location_text + ' '



        if self.include_agent_action:
            text += 'Your teammates action in last round: '
            for a in initial_actions.keys():
                if initial_actions[a].name == 'remove_bomb_beacon':
                    text += 'Player {id}: "{action}". '.format(id=a, action='Sent a communication message to the Team.')
                elif initial_actions[a].name == 'remove_help_beacon':
                    text += 'Player {id}: "{action}". '.format(id=a, action='Invalid action.')
                else:
                    text += 'Player {id}: "{action}". '.format(id=a, action=initial_actions[a].name.replace('_', ' '))

        if self.allow_comm:
            text += 'Communication messages sent by your teammates: '
            for a in communications.keys():
                text += 'Player {id}: "{comm}". '.format(id=a, comm=communications[a])



        # print(text)
        return obs, reward, done, info, text

    def step_text(self,agent_id, round, initial_actions, communications):
        # action is object
        # agent is str index


        action = initial_actions[agent_id]
        obs_text = 'Your action is invalid.'

        Action = self.env.action_enum
        agent = self.env.agents[agent_id]
        if action is None:
            obs_text = "Invalid action."
        elif action == Action.inspect_bomb:
            if agent.bomb:
                bomb_id = str(agent.bomb.id)
                sequence = '-'.join([COLOR_TO_STR[x] for x in agent.bomb._full_sequence[agent.bomb._current_step:]])
                num_stage = str(agent.bomb.num_stages)
                obs_text = "You inspected Bomb {bomb_id}. This bomb is a {num_stage}-stage bomb and its remaining sequence is {sequence}.".format(
                    bomb_id=bomb_id, num_stage=num_stage, sequence=sequence)

            else:
                current_room = str(agent.node.id)
                obs_text = "There is no bomb in the current current location, Room {current_room}, for you to inspect.".format(
                    current_room=current_room)
        elif action.node() is not None:
            # Go to the given node
            if self.env._get_action_mask(agent_id)[action] or action.node().id == agent.node.id:
                room_id = str(action.node().id)
                room_agents = ' and '.join([str(x.id) for x in action.node().agents])
                room_bombs = ' and '.join(['Bomb ' + str(x.id) for x in action.node().bombs])
                obs_text = "You moved to Room {room_id}. In the new room you found {room_agents}, {room_bombs}.".format(
                    room_id=room_id, room_agents=room_agents, room_bombs=room_bombs)

            else:
                room_id = str(action.node().id)
                current_room = str(agent.node.id)
                obs_text = "You can not directly move to Room {room_id} because it is not adjacent to your current location, Room {current_room}. Consider taking a detour to another room first and then move to your destination.".format(
                    room_id=room_id, current_room=current_room)


        elif action.tool() is not None:
            tool = agent.get_tool_from_inventory(action.tool())
            if tool in Tool.bomb_tools():
                if agent.bomb:
                    if tool.color == agent.bomb.color:

                        tool_color = COLOR_TO_STR[tool.color]
                        bomb_id = str(agent.bomb.id)
                        sequence = '-'.join(
                            [COLOR_TO_STR[x] for x in agent.bomb._full_sequence[agent.bomb._current_step+1:]])

                        obs_text = "You applied Tool {tool_color} to Bomb {bomb_id}. This bomb has a remaining sequence of {sequence}.".format(
                            tool_color=tool_color, bomb_id=bomb_id, sequence=sequence)

                    else:
                        tool_color = COLOR_TO_STR[tool.color]
                        bomb_id = agent.bomb.id
                        sequence = agent.bomb._full_sequence[agent.bomb._current_step:]


                        obs_text = "You can not apply Tool {tool_color} to Bomb {bomb_id} because the sequence of this bomb is {sequence}. You will need to apply other color tool first.".format(
                            tool_color=tool_color, bomb_id=bomb_id, sequence=sequence)
                else:
                    obs_text = "There is no bomb in your current location, room {room_id}, for you to defuse.".format(
                        room_id=agent.node.id)
            else:
                obs_text = "You do not have {tool}. Consider asking your teammates who have this tool to help you defuse the bomb.".format(
                    tool=COLOR_TO_STR[action.tool()])





        # team_location_text = 'Player alpha is in Room {loc_a}; Player bravo is in Room {loc_b}; Player charlie is in Room {loc_c}.'.format(
        #     loc_a=str(self.env.agents['alpha'].node.id), loc_b=str(self.env.agents['bravo'].node.id),
        #     loc_c=str(self.env.agents['charlie'].node.id))

        room_id = str(agent.node.id)
        room_agents = ' and '.join([str(x.id) for x in agent.node.agents])
        room_bombs = ' and '.join(['Bomb ' + str(x.id) for x in agent.node.bombs])
        room_contents = "You are currently in Room {room_id}. Contents of this room include {room_agents}, {room_bombs}.".format(
            room_id=room_id, room_agents=room_agents, room_bombs=room_bombs)

        text = 'Your observation is: '
        text += 'Round: {timestep}. '.format(timestep=str(round))
        text += 'Total team score: {score}. '.format(score=str(self.env.score))
        text += 'Results: ' + obs_text + ' '
        text += 'Room contents: '+ room_contents+ ' '
        # text += 'Teammate Locations: ' + team_location_text + ' '



        if self.include_agent_action:
            text += 'Your teammates action in last round: '
            for a in initial_actions.keys():
                if initial_actions[a].name == 'remove_bomb_beacon':
                    text += 'Player {id}: "{action}". '.format(id=a, action='Sent a communication message to the Team.')
                elif initial_actions[a].name == 'remove_help_beacon':
                    text += 'Player {id}: "{action}". '.format(id=a, action='Invalid action.')
                else:
                    text += 'Player {id}: "{action}". '.format(id=a, action=initial_actions[a].name.replace('_', ' '))

        if self.allow_comm:
            text += 'Communication messages sent by your teammates: '
            for a in communications.keys():
                text += 'Player {id}: "{comm}". '.format(id=a, comm=communications[a])



        # print(text)
        return text

    def decode_action(self, chat_output):

        Action = self.env.action_enum
        comm = ''
        action = None
        if self.act_and_comm:
            if len(chat_output.split('Message to Team:')) > 1 and len(chat_output.split('"')) > 1:
                comm = chat_output.split('Message to Team:')[1].split('"')[1]
                chat_output = chat_output.split('Message to Team:')[0]+ chat_output.split('Message to Team:')[1].split('"')[2]
            else:
                comm = ''
            if 'inspect' in chat_output.lower():
                action = Action.inspect_bomb
            elif 'move to room' in chat_output.lower():
                room_id = int(chat_output.lower().split('move to room')[1][1])
                if room_id in self.valid_node:
                    action = Action.go_to(room_id)
                else:
                    action = None
            elif 'go_to_node_' in chat_output.lower():
                room_id = int(chat_output.lower().split('go_to_node_')[1][0])
                action = Action.go_to(room_id)
            elif 'apply' in chat_output.lower() or 'defuse' in chat_output.lower():
                if 'red' in chat_output.lower():
                    action = Action.use_tool(Tool.red)
                elif 'blue' in chat_output.lower():
                    action = Action.use_tool(Tool.blue)
                elif 'green' in chat_output.lower():
                    action = Action.use_tool(Tool.green)
            # elif 'wait' in chat_output.lower() or 'stay' in chat_output.lower():
            #     action = Action.remove_help_beacon
            else:
                action = None
                # action, comm = self.decode_action_API(chat_output, comm = comm)

        return action, comm






    def load(self,saved_files,ending_round=999):
        Action = self.env.action_enum
        # saved_files = {'alpha': DATA_PATH + 'gpt-4_0.7_alpha_05-25-19-18-01.json',
        #                'bravo': DATA_PATH + 'gpt-4_0.7_bravo_05-25-19-19-18.json',
        #                'charlie': DATA_PATH + 'gpt-4_0.7_charlie_05-25-19-20-49.json'}

        chat_agents = {'alpha': ChatAgent(agent_id='alpha'), 'bravo': ChatAgent(agent_id='bravo'),
                       'charlie': ChatAgent(agent_id='charlie')}
        initial_actions = {'alpha': Action.go_to(0), 'bravo': Action.go_to(0), 'charlie': Action.go_to(0)}
        communications = {'alpha': 'None', 'bravo': 'None', 'charlie': 'None'}

        round = 1
        record = saved_files['record']
        with open(record, 'r', encoding='utf-8') as f:
            data = f.read()
            new_data = data.replace('}{', '},{')
            json_data = json.loads(f'[{new_data}]')
            print(json_data)


        for r in json_data:
            if 'action' in r.keys():
                agent_id = r['agent_id']
                initial_actions[agent_id] = Action[r['action'].replace(' ', '_')]
                communications[agent_id] = r['comm']
                _, reward, done, info, obs_text = self.step(agent_id, round, initial_actions, communications)
                round = r['round']
            else:
                agent_id = r['agent_id']
                initial_actions[agent_id], communications[agent_id] = self.decode_action(r["chat_output"])
                _, reward, done, info, obs_text = self.step(agent_id, round, initial_actions, communications)
                round = r['round']
            if round >= ending_round:
                break



        for agent_id in ['alpha','bravo','charlie']:

            saved_file = saved_files[agent_id]
            with open(saved_file, 'r', encoding='utf-8') as f:
                file = json.load(f)
            chat_agents[agent_id] = ChatAgent(agent_id=file['agent_id'],model=file['model'],temperature=file['temperature'],message_history=file['message_history'],belief=True,allow_comm=True)

            initial_actions[agent_id],communications[agent_id] = self.decode_action(file['message_history'][-2]['content'])

        round += 1

        return chat_agents, initial_actions, communications, round


    def to_csv(self,saved_file,output_path):
        Action = self.env.action_enum
        initial_actions = {'alpha': Action.go_to(0), 'bravo': Action.go_to(0), 'charlie': Action.go_to(0)}
        communications = {'alpha': 'None', 'bravo': 'None', 'charlie': 'None'}

        output = []
        record = saved_file
        with open(record, 'r', encoding='utf-8') as f:
            data = f.read()
            new_data = data.replace('}{', '},{')
            json_data = json.loads(f'[{new_data}]')
            print(json_data)

        for r in json_data:

            agent_id = r['agent_id']
            initial_actions[agent_id], communications[agent_id] = self.decode_action(r["chat_output"])
            _, reward, done, info, obs_text = self.step(agent_id, initial_actions, communications)
            round = r['round']
            row = [round,agent_id,initial_actions[agent_id],communications[agent_id],reward,done,obs_text]
            output.append(row)

        csv = pd.DataFrame(output, columns=["round","agent_id","action","comm","reward","done","obs_text"])
        csv.to_csv(output_path)



MAX_RETRIES = 10
RETRY_DELAY = 3


BACKGROUND_PROMPT_NEW = "Welcome to our interactive text game! In this game, you'll assume the role of a specialist on a search and rescue team. Alongside two other players, you'll navigate a five-room environment with a mission to defuse five hidden bombs. Your call sign is {agent_id}\
The Map: Imagine a network of rooms represented by a connected graph where each node corresponds to a room, and the edges between nodes depict hallways. The rooms are numbered 0, 3, 6, 5, and 8. Room 0 is connected to all other rooms. Room 5 shares a hallway with room 6. Room 3 is linked to room 8. And room 8 is also connected with room 6. You can only travel to adjacent, directly connected rooms at each turn.\
The Challenge: Scattered among these rooms are five bombs, each coded with different phases represented by colors. To defuse them, you'll need to use the correct wire-cutting tools in the correct sequence. There are one-phase, two-phase, and three-phase bombs, needing 1, 2, or 3 color-coded tool applications in sequence to disarm. For instance, a bomb with a red-green phase sequence requires the red tool first, then the green one. Points are awarded based on the number of tools used for defusing a bomb, with each tool use worth 10 points. Your task is to maximize the team score as soon as possible. The challenge is that the bomb locations and sequences are unknown to players at the start.\
Tools: Each player is equipped with two color-coded wire cutters. Player Alpha has red and green tools, player Bravo wields green and blue, and player Charlie possesses blue and red.\
Actions: Each round, you can opt to do one of the following: 1) Move to an adjacent room, 2) Inspect a bomb's phase sequence in your current room, or 3) Apply your wire cutters to a bomb in the current room. \
Communications: In addition to selecting an action to take from the above list, you can also send communication message texts to both of your teammates in each round. The message text you sent will be shared with both of your teammates in their observation in the next round. \
Observation: While you can only see what's in your current room and read text messages from teammates. You'll also be informed of the current round number, team score and the current location of your teammates. Your teammates have the same observability as you. They will not be able to know your action and its consequences unless you explicitly communicate. \
You will be playing as Player {agent_id}. To facilitate our interaction, reply your action selection and communication messages in this fixed format: Action selection: Your action. Message to Team: “Your Message”. To move to an adjacent room, say: 'Move to Room X'. To inspect the sequence of a bomb in your current room, say: 'Inspect Bomb'. To apply a wire cutter tool, say: 'Apply X Tool'. Remember, your replies must adhere strictly to these rules. Feel free to ask clarifying questions if needed. I'll supply the necessary information as we progress. Are you ready to take on this explosive challenge?"

BACKGROUND_PROMPT_NOCOMM = "Welcome to our interactive text game! In this game, you'll assume the role of a specialist on a search and rescue team. Alongside two other players, you'll navigate a five-room environment with a mission to defuse five hidden bombs. Your call sign is {agent_id}\
The Map: Imagine a network of rooms represented by a connected graph where each node corresponds to a room, and the edges between nodes depict hallways. The rooms are numbered 0, 3, 6, 5, and 8. Room 0 is connected to all other rooms. Room 5 shares a hallway with room 6. Room 3 is linked to room 8. And room 8 is also connected with room 6. You can only travel to adjacent, directly connected rooms at each turn.\
The Challenge: Scattered among these rooms are five bombs, each coded with different phases represented by colors. To defuse them, you'll need to use the correct wire-cutting tools in the correct sequence. There are one-phase, two-phase, and three-phase bombs, needing 1, 2, or 3 color-coded tool applications in sequence to disarm. For instance, a bomb with a red-green phase sequence requires the red tool first, then the green one. Points are awarded based on the number of tools used for defusing a bomb, with each tool use worth 10 points. Your task is to maximize the team score as soon as possible. The challenge is that the bomb locations and sequences are unknown to players at the start.\
Tools: Each player is equipped with two color-coded wire cutters. Player Alpha has red and green tools, player Bravo wields green and blue, and player Charlie possesses blue and red.\
Actions: Each round, you can opt to do one of the following: 1) Move to an adjacent room, 2) Inspect a bomb's phase sequence in your current room, or 3) Apply your wire cutters to a bomb in the current room. \
Observation: While you can only see what's in your current room. You'll also be informed of the current round number and team score. Your teammates have the same observability as you. They will not be able to know your action and its consequences unless you are in the same room. \
You will be playing as Player {agent_id}. To facilitate our interaction, reply your action selection in this fixed format: Action selection: Your action. To move to an adjacent room, say: 'Move to Room X'. To inspect the sequence of a bomb in your current room, say: 'Inspect Bomb'. To apply a wire cutter tool, say: 'Apply X Tool'. Remember, your replies must adhere strictly to these rules. Feel free to ask clarifying questions if needed. I'll supply the necessary information as we progress. Are you ready to take on this explosive challenge?"

# BACKGROUND_PROMPT = "Welcome to our interactive text game! In this game, you'll assume the role of a specialist on a search and rescue team. Alongside two other players, you'll navigate a five-room environment with a mission to defuse five hidden bombs. The Map: Imagine a network of rooms represented by a connected graph where each node corresponds to a room, and the edges between nodes depict hallways. The rooms are numbered 0, 3, 6, 5, and 8. Room 0 is connected to all other rooms. Room 5 shares a hallway with room 6, room 3 is linked to room 8, and room 8 is also connected with room 6. You can only travel to adjacent, directly connected rooms each turn. The Challenge: Scattered among these rooms are five bombs, each coded with different phases represented by colors. To defuse them, you'll need to use the correct wire-cutting tools in the correct sequence. There are one-phase, two-phase, and three-phase bombs, needing 1, 2, or 3 color-coded tool applications in sequence to disarm. For instance, a bomb with a red-green phase sequence requires the red tool first, then the green one. The challenge is that the bomb locations and sequences are unknown to players at the start. Your Tools: Each player is equipped with two color-coded wire cutters. As player Alpha, you have red and green tools, player Bravo wields green and blue, and player Charlie possesses blue and red. Your Actions: Each round, you can opt to do one of the following: 1) Move to an adjacent room, 2) Inspect a bomb's phase sequence in your current room, 3) Apply your wire cutters to a bomb, or 4) send text messages to both of your teammates. Observation: While you can only see what's in your current room and read text messages from teammates. You'll also be informed of the current round number. Any successful bomb defusal, along with the accumulated team score, is broadcasted to all players. Points are awarded based on the number of tools used for defusing a bomb, with each tool use worth 10 points. Remember, your actions must adhere strictly to these rules. Feel free to ask clarifying questions if needed. I'll supply the necessary information as we progress. You will be playing as Player {agent_id}. Are you ready to take on this explosive challenge?"
# INSTRUCT_PROMPT = "Your current observation is: Round: 1. Total team score: 0. Observation: You are currently in Room 0 with both of your teammates. In the room you also found bomb 1 with unknown sequence. There is no other bomb in the current room. Teammate Locations: Player alpha is in Room 0; Player bravo is in Room 0; Player charlie is in Room 0. Communication messages sent by your teammates: Player alpha: "". Player bravo: "". Player charlie: "". Given the above observation, what is your next action?"

INITIAL_BELIEF = "Below is your current belief about game state based on your previous observations about the environment and interactions with your teammates. Your role: You are playing as Player {agent_id}.\n Current round: 0.\n Total team score: 0.\n Restuls: You moved to Room {initial_node}. In the new room you found Player alpha, Player bravo, Player charlie, and Bomb {initial_bomb}.\n Room Contents: You are currently in Room {initial_node} with both of your teammates. In the room you also found bomb {initial_bomb} with unknown sequence. There is no other bomb in the current room.\n Teammate Locations: Player alpha is in Room {initial_node}; Player bravo is in Room {initial_node}; Player charlie is in Room {initial_node}.\n Room connectivity: Room 0 is connected to room 3, 5, 6,8. Room 3 is connected to room 0. Room 5 is connected to room 0 and 6. Room 6 is connected to room 0 and 8. Room 8 is connected to room 0 and 6.\n Bomb Intel: Bomb {initial_bomb}: Located in Room {initial_node}. The phase sequence is Unknown. Ohter bomb details currently unknown.\n Communication messages sent by your teammates: Player alpha: None. Player bravo: None. Player charlie: None.\n Tool inventory: Alpha: Equipped with red and green wire cutters. Bravo: Equipped with green and blue wire cutters. Charlie: Equipped with red and blue wire cutters.\n Available action options: 1. To move to an adjacent room, say: 'Move to Room X'. 2. To inspect the sequence of a bomb in your current room, say: 'Inspect Bomb'. 3. To apply a wire cutter tool, say: 'Apply X Tool'. 4. To send a message to your teammates, say: 'Message to Team: \"Your Message\"'."

INITIAL_BELIEF_NOCOMM = "Below is your current belief about game state based on your previous observations about the environment and interactions with your teammates. Your role: You are playing as Player {agent_id}.\n Current round: 0.\n Total team score: 0.\n Restuls: You moved to Room {initial_node}. In the new room you found Player alpha, Player bravo, Player charlie, and Bomb {initial_bomb}.\n Room Contents: You are currently in Room {initial_node} with both of your teammates. In the room you also found bomb {initial_bomb} with unknown sequence. There is no other bomb in the current room.\n Teammate Locations: Player alpha is in Room {initial_node}; Player bravo is in Room {initial_node}; Player charlie is in Room {initial_node}.\n Room connectivity: Room 0 is connected to room 3, 5, 6,8. Room 3 is connected to room 0. Room 5 is connected to room 0 and 6. Room 6 is connected to room 0 and 8. Room 8 is connected to room 0 and 6.\n Bomb Intel: Bomb {initial_bomb}: Located in Room {initial_node}. The phase sequence is Unknown. Ohter bomb details currently unknown.\n Tool inventory: Alpha: Equipped with red and green wire cutters. Bravo: Equipped with green and blue wire cutters. Charlie: Equipped with red and blue wire cutters.\n Available action options: 1. To move to an adjacent room, say: 'Move to Room X'. 2. To inspect the sequence of a bomb in your current room, say: 'Inspect Bomb'. 3. To apply a wire cutter tool, say: 'Apply X Tool'. 4."



UPDATE_PROMPT = "Please update your belief state based on the above observation, and send me your results in the same format as previously. DO NOT add additional explanations. Return your answer begining with: 'Below is your current belief'"


INITIAL_PROMPT = "Given the above belief state, what is your next action?"

class ChatAgent():
    def __init__(self,agent_id='alpha',model="gpt-4-turbo-preview",temperature=0.0,message_history =None, belief = False, allow_comm = True,initial_bomb = 1, initial_node = 0):
        self.agent_id = agent_id
        self.model = model
        self.temperature=temperature
        self.belief = belief
        # self.last_belief = INITIAL_BELIEF.format(agent_id = agent_id,initial_bomb = initial_bomb,initial_node=initial_node)
        self.allow_comm = allow_comm
        if self.allow_comm:
            BACKGROUND_PROMPT = BACKGROUND_PROMPT_NEW
            self.last_belief = INITIAL_BELIEF.format(agent_id = agent_id,initial_bomb = initial_bomb,initial_node=initial_node)
        else:
            BACKGROUND_PROMPT = BACKGROUND_PROMPT_NOCOMM
            self.last_belief = INITIAL_BELIEF_NOCOMM.format(agent_id=agent_id, initial_bomb=initial_bomb,
                                                    initial_node=initial_node)
        if message_history is None:
            self.message_history = [
                {"role": "system", "content": 'You are playing a text game with the user.'},
                {"role": "user", "content": BACKGROUND_PROMPT.format(agent_id = agent_id)},
                # {"role": "user", "content": INSTRUCT_PROMPT},
            ]
            if self.belief:
                self.message_history.append({"role": "user", "content": self.last_belief+INITIAL_PROMPT})
        else:
            self.message_history = message_history

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def makeAPIcall(self):


        response = openai.chat.completions.create(
                    model=self.model,
                    messages=self.message_history,
                    temperature=self.temperature
                )
        return response.choices[0].message.content


    def update_history(self,text):

        if self.belief:
            text += UPDATE_PROMPT
            if self.model == 'gpt-3.5-turbo':
                last_action = self.message_history.pop()['content']
                self.message_history.append({"role": "user", "content": last_action+ text})
                new_belief = self.makeAPIcall()
                self.message_history.pop()
                self.message_history.append({"role": "assistant", "content": last_action})
                self.message_history.append({"role": "user", "content": 'Below is your current belief about game state based on your previous observations about the environment and interactions with your teammates. '+new_belief+INITIAL_PROMPT})
            else:
                self.message_history.append({"role": "user", "content": text})
                new_belief = self.makeAPIcall()
                self.message_history.pop()
                self.message_history.append({"role": "user", "content": new_belief+INITIAL_PROMPT})

            # new_belief = self.update_belief(text)
            # self.message_history.append({"role": "user", "content": new_belief+INITIAL_PROMPT})

        else:

            text += "Given the above observation, what is your next action?"
            print('env', text)
            self.message_history.append({"role": "user", "content": text})
            new_belief = "None"



        # if len(self.message_history)>=7:
        #     self.message_history.pop(2)
        #     self.message_history.pop(2)
        return new_belief

    def step(self):
        action = self.makeAPIcall()

        self.message_history.append({"role": "assistant", "content": action})
        print(self.agent_id,action)
        return action

    def save(self,data_path):
        data = {}
        data['agent_id'] = self.agent_id
        data['model'] = self.model
        data['temperature'] = self.temperature
        data['message_history'] = self.message_history
        data['belief'] = self.belief
        data['allow_comm'] = self.allow_comm
        timestr = time.strftime("%m-%d-%H-%M-%S", time.localtime())
        path = data_path + '{model}_{temperature}_{allow_comm}_{agent_id}_{timestr}.json'.format(timestr=timestr,agent_id = self.agent_id,allow_comm = self.allow_comm,model = self.model,temperature = self.temperature)
        with open(path, 'w+', encoding='utf-8') as f:
            json.dump(data, f)

    def ask(self, question_text):
        self.message_history.append({"role": "user", "content": question_text})
        reply = self.makeAPIcall()
        self.message_history.pop()
        return reply


    def read_belief(self):
        return self.message_history
