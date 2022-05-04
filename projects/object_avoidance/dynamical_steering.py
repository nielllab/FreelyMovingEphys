

class DynamicalSteeringModel:
    def __init__(self, time, agent_x, agent_y, goal_x, goal_y, obstacle_x, obstacle_y):
        self.num_y_bins =  40
        self.agent_x = agent_x
        self.agent_y = agent_y
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.obstacle_x = obstacle_x
        self.obstacle_y = obstacle_y
        self.time = time

        self.goal_distance
        self.obstacle_distance
        self.turning_rate
        self.goal_attraction
        self.obstacle_repulsion
        self.b # dampening coefficient

    def calc_props(self):
        # goal distance

        # obstacle distance

        # turning rate

        # goal attraction

        # obstacle repulsion

        # dampening coeffeficient

        # agent speed
        self.agent_speed = np.sqrt((np.diff(self.agent_x)/self.time)**2 + (np.diff(self.agent_y)/self.time)**2)
        # heading, angle to goal, angle to obstacle
        self.heading = np.zeros(len(self.agent_x))
        self.goal_ang = np.zeros(len(self.agent_x))
        self.obstacle_ang = np.zeros(len(self.agent_x))
        for frame in range(len(self.agent_x)):
            self.heading = np.arctan((self.agent_x[frame] - self.agent_x[frame-1]) / (self.agent_y[frame] - self.agent_y[frame-1]))
            self.goal_ang = np.arctan((self.goal_x[frame] - self.agent_x[frame]) / (self.goal_y[frame] - self.agent_y[frame]))
            self.obstacle_ang = np.arctan((self.obstacle_x[frame] - self.agent_x[frame]) / (self.obstacle_y[frame] - self.agent_y[frame]))
        self.goal_ang_wrt_heading = self.heading - self.goal_ang
        self.obstacle_ang_wrt_heading = self.heading - self.obstacle_ang

    def calc_paths(self):
        # 
        self.agent_x_binned, self.agent_x_bin_edges = np.digitize(self.agent_x, bins=self.num_y_bins)

    def calc_ang_acc(self):
        self.ang_acc = -self.b*self.turning_rate - self.goal_attraction(self.heading - self.goal_direction) + self.obstacle_repulsion(self.heading - self.obstacle_direction) * np.exp(-self.heading - self.obstacle_direction)



