from .dragon import DragonBaseEnv
from .core import Agent, Bomb, BombBeacon, Color, Tool

from BaseAgent import MissionContext
from MinecraftBridge.messages import (
    CommunicationEnvironment,
    ItemStateChange,
    ObjectStateChange,
    PlayerInventoryUpdate,
    PlayerState,
    PlayerStateChange,
    ScoreUpdate,
    Trial,
)



TOOLS = {
    'BOMB_DISPOSER': Tool.disposer,
    'BOMB_PPE': Tool.ppe,
    'BOMB_SENSOR': Tool.sensor,
    'FIRE_EXTINGUISHER': Tool.fire_extinguisher,
    'WIRECUTTERS_RED': Tool.red,
    'WIRECUTTERS_GREEN': Tool.green,
    'WIRECUTTERS_BLUE': Tool.blue,
}



class DragonBridgeEnv(DragonBaseEnv):
    """
    Dragon environment that updates via messages from MinecraftBridge.
    """

    def __init__(self, context: MissionContext, **kwargs):
        super().__init__(**kwargs)
        self.context = context
        self.bomb_map = {}
        self.pending_bomb_beacons = set()
        self.pending_bomb_dependencies = {}

        context.minecraft_interface.register_callback(
            CommunicationEnvironment, self.__onCommunicationEnvironment)
        context.minecraft_interface.register_callback(
            ItemStateChange, self.__onItemStateChange)
        context.minecraft_interface.register_callback(
            ObjectStateChange, self.__onObjectStateChange)
        context.minecraft_interface.register_callback(
            PlayerInventoryUpdate, self.__onPlayerInventoryUpdate)
        context.minecraft_interface.register_callback(
            PlayerState, self.__onPlayerState)
        context.minecraft_interface.register_callback(
            PlayerStateChange, self.__onPlayerStateChange)
        context.minecraft_interface.register_callback(
            ScoreUpdate, self.__onScoreUpdate)
        context.minecraft_interface.register_callback(
            Trial, self.__onTrial)

        self.reset()
    
    def add_bomb(self, bomb_id, location, sequence, fuse):
        bomb = Bomb(
            bomb_id=bomb_id,
            location=(int(location[0]), int(location[2])),
            sequence=(Color.from_char(c) for c in sequence),
            fuse=int(fuse) * 60,
            on_state_change=self._on_bomb_state_change,
        )

        self.bomb_map[bomb.id] = bomb

        self._bombs.append(bomb)
        self._node_grid[bomb.location].add_child(bomb) # update graph
        self._block_grid[bomb.location] = 'bomb_inactive' # update grid

        return bomb
    
    def __onCommunicationEnvironment(self, msg):
        self.tick(dt=(msg.elapsed_milliseconds / 1000 - self.time))

        try:
            bomb_info = dict(line.split(': ') for line in msg.message.split('\n'))
        except:
            return

        if bomb_info['CHAINED_ID'] == 'NONE':
            dependency = None
        elif bomb_info['CHAINED_ID'] in self.bomb_map:
            dependency = self.bomb_map[bomb_info['CHAINED_ID']]
        else:
            dependency = None
            self.pending_bomb_dependencies[bomb_info['CHAINED_ID']] = bomb_info['BOMB_ID']
        
        if bomb_info['BOMB_ID'] not in self.bomb_map:
            bomb = self.add_bomb(
                bomb_id=bomb_info['BOMB_ID'],
                location=msg.sender_position,
                sequence=bomb_info['SEQUENCE'].strip('[]').split(', '),
                fuse=bomb_info['FUSE_START_MINUTE'],
            )
        
        bomb = self.bomb_map[bomb_info['BOMB_ID']]
        bomb._dependency = dependency

        if bomb.id in self.pending_bomb_beacons:
            self.pending_bomb_beacons.remove(bomb.id)
            self._node_grid[bomb.location].add_child(bomb)

        if bomb.id in self.pending_bomb_dependencies:
            dependent_bomb = self.bomb_map[self.pending_bomb_dependencies[bomb.id]]
            dependent_bomb._dependency = bomb

    def __onItemStateChange(self, msg):
        self.tick(dt=(msg.elapsed_milliseconds / 1000 - self.time))
        agent = self.agents[msg.owner]
        if msg.item_name in TOOLS:
            agent.get_tool_from_inventory(TOOLS[msg.item_name])
        elif msg.item_name == 'HAZARD_BEACON':
            agent.node.add_child(agent.create_help_beacon())
        elif msg.item_name == 'BOMB_BEACON':
            bomb_id = msg.current_attributes.get_attribute('bomb_id')
            if bomb_id in self.bomb_map:
                agent.node.add_child(BombBeacon(self.bomb_map[bomb_id]))
            else:
                self.pending_bomb_beacons.add(bomb_id)

    def __onObjectStateChange(self, msg):
        self.tick(dt=(msg.elapsed_milliseconds / 1000 - self.time))

        if msg.id not in self.bomb_map:
            self.add_bomb(
                bomb_id=msg.id,
                location=msg.location,
                sequence=msg.current_attributes.get_attribute('sequence'),
                fuse=msg.current_attributes.get_attribute('fuse_start_minute'),
            )

        bomb = self.bomb_map[msg.id]

        # Update bomb type
        if msg.type == 'block_bomb_fire':
            bomb._is_fire_bomb = True

        # Update bomb sequence
        sequence = msg.changed_attributes.get_attribute('sequence')
        if sequence and len(sequence.current) != len(bomb.sequence):
            bomb._current_step += 1

        # Update bomb state
        outcome = msg.changed_attributes.get_attribute('outcome')
        if outcome:
            if outcome.current == 'INACTIVE':
                bomb.state = Bomb.BombState.inactive
            elif outcome.current in {'TRIGGERED', 'TRIGGERED_ADVANCE_SEQ'}:
                bomb.state = Bomb.BombState.active
            elif outcome.current in {'EXPLODE_TOOL_MISMATCH', 'EXPLODE_TIME_LIMIT', 'EXPLODE_FIRE'}:
                bomb.state = Bomb.BombState.exploded
            elif outcome.current == 'DEFUSED':
                bomb._value = Bomb.value_per_stage * bomb.num_stages
                bomb.state = Bomb.BombState.defused
            elif outcome.current == 'DEFUSED_DISPOSER':
                bomb._value = Bomb.value_per_stage
                bomb.state = Bomb.BombState.defused

    def __onPlayerInventoryUpdate(self, msg):
        self.tick(dt=(msg.elapsed_milliseconds / 1000 - self.time))

        for agent_id, agent_inventory in msg.current_inventory.items():
            for item, quantity in agent_inventory.items():
                if item in TOOLS:
                    tool = TOOLS[item]
                    agent = self.agents[agent_id]
                    agent.assign_tool(
                        tool,
                        quantity=max(0, quantity - agent.tool_remaining_uses[tool]),
                    )

    def __onPlayerState(self, msg):
        self.tick(dt=(msg.elapsed_milliseconds / 1000 - self.time))
        x, z = int(msg.x), int(msg.z)
        if self._node_grid.is_in_bounds((x, z)):
            node = self._node_grid[x, z]
            if node:
                self.agents[msg.participant_id].go_to(node)

    def __onPlayerStateChange(self, msg):
        self.tick(dt=(msg.elapsed_milliseconds / 1000 - self.time))
        agent = self.agents[msg.participant_id]
        agent._health = float(msg.current_attributes.get_attribute('health')) / 20

    def __onScoreUpdate(self, msg):
        self._score = msg.teamScore

    def __onTrial(self, msg):
        if len(self.context.participants) > 0:
            # Reset agents
            self.agents.clear()
            for participant in self.context.participants:
                self.agents[participant.id] = Agent(participant.id)

            # Reinitialize graph
            self._init_graph(
                centroids={node.id: node.centroid for node in self.graph.nodes.values()},
                edges=self.graph.edges,
                valid_regions={node.region for node in self.graph.nodes.values()},
            )

            # Repopulate node grid
            for loc in self._node_grid.locations():
                if self._node_grid[loc]:
                    self.graph.nodes[self._node_grid[loc].id].area = self._node_grid[loc].area
                    self._node_grid[loc] = self.graph.nodes[self._node_grid[loc].id]

            # Clear observations (hacky)
            self.observations.clear()

            self.bomb_map.clear()

            self.reset(num_bombs_per_region=0)
