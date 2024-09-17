import numpy as np
import random
import matplotlib.pyplot as plt
import heapq
import copy

RIGHT_ARROW = '\u2192'
LEFT_ARROW = '\u2190'
UP_ARROW = '\u2191'
DOWN_ARROW = '\u2193'

def parse_layout(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    grid = []
    agent_pos = None
    goal_pos = None

    for x, line in enumerate(lines):
        row = list(line.strip())
        grid.append(row)
        for y, cell in enumerate(row):
            if cell == 'A':
                agent_pos = (x, y)
            elif cell == 'G':
                goal_pos = (x, y)

    return grid, agent_pos, goal_pos
def initialize_q_table(grid):
    rows, cols = len(grid), len(grid[0])
    q_table = np.ones((rows, cols, 4))  # 4 actions: up, down, left, right
    return q_table

def choose_action(state, q_table, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice([0, 1, 2, 3])  # Random action
    else:
        return np.argmax(q_table[state[0], state[1]])  # Best action

def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    max_next_q = np.max(q_table[next_state[0], next_state[1]])
    q_table[state[0], state[1], action] += alpha * (reward + gamma * max_next_q - q_table[state[0], state[1], action])

def find_shortest_path(q_table, start, goal, grid):
    path = []
    actions = []
    state = start



    while state != goal:
        path.append(state)
        x, y = state
        action = np.argmax(q_table[x, y])

        if (action == 0):
            actions.append(UP_ARROW)
        elif(action == 1):
            actions.append(DOWN_ARROW)
        elif(action == 2):
            actions.append(LEFT_ARROW)
        else:
            actions.append(RIGHT_ARROW)

        state = perform_action(state, action, grid)

    path.append(goal)
    return path, actions

def q_learning(grid, agent_pos, goal_pos, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    flag_first_time = 0
    q_table = initialize_q_table(grid)
    rows, cols = len(grid), len(grid[0])
    steps_per_episode = []

    for episode in range(episodes):
        state = agent_pos
        steps = 0

        while state != goal_pos:
            action = choose_action(state, q_table, epsilon)
            next_state = perform_action(state, action, grid)
            reward = 1 if next_state == goal_pos else 0

            update_q_table(q_table, state, action, reward, next_state, alpha, gamma)
            state = next_state
            steps += 1
        steps_per_episode.append(steps)
        if(steps == 14):
            if(flag_first_time == 0):
                first_time_optimal_policy = episode
                print("First time reaching optimal policy: ", episode)
                flag_first_time = 1

    return q_table, steps_per_episode, first_time_optimal_policy

def perform_action(state, action, grid):
    x, y = state
    if action == 0:  # Up
        x = max(0, x - 1)
    elif action == 1:  # Down
        x = min(len(grid) - 1, x + 1)
    elif action == 2:  # Left
        y = max(0, y - 1)
    elif action == 3:  # Right
        y = min(len(grid[0]) - 1, y + 1)

    if grid[x][y] == 'O':  # Hit an obstacle
        return state

    return (x, y)

def dyna_q(grid, agent_pos, goal_pos, n_planning_steps, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    flag_first_time = 0
    q_table = initialize_q_table(grid)
    rows, cols = len(grid), len(grid[0])
    model = {}
    steps_per_episode = []

    for episode in range(episodes):
        state = agent_pos
        steps = 0

        while state != goal_pos:
            action = choose_action(state, q_table, epsilon)
            next_state = perform_action(state, action, grid)
            reward = 1 if next_state == goal_pos else 0

            update_q_table(q_table, state, action, reward, next_state, alpha, gamma)

            # Model update
            model[(state, action)] = (reward, next_state)

            # Planning
            for _ in range(n_planning_steps):
                state_prime, action_prime = random.choice(list(model.keys()))
                reward_prime, next_state_prime = model[(state_prime, action_prime)]
                update_q_table(q_table, state_prime, action_prime, reward_prime, next_state_prime, alpha, gamma)

            state = next_state
            steps += 1
        steps_per_episode.append(steps)
        if (steps == 14):
            if (flag_first_time == 0):
                first_time_optimal_policy = episode
                print("First time reaching optimal policy: ", episode)
                flag_first_time = 1

    return q_table, steps_per_episode, first_time_optimal_policy

def dyna_q_experiment(file_path1, file_path2, n_planning_steps=50, alpha=0.1, gamma=0.9, epsilon=0.1):


    grid1, agent_pos1, goal_pos1 = parse_layout(file_path1)
    grid2, agent_pos2, goal_pos2 = parse_layout(file_path2)
    flag_change = 0

    current_agent_pos = agent_pos1
    current_goal_pos = goal_pos1
    current_grid = grid1

    q_table = initialize_q_table(current_grid)
    #rows, cols = len(grid), len(grid[0])
    model = {}
    steps_per_episode = []

    min_steps_map1 = np.inf
    min_steps_map2 = np.inf

    while True:
        state = current_agent_pos
        steps = 0

        while state != current_goal_pos:
            action = choose_action(state, q_table, epsilon)
            next_state = perform_action(state, action, current_grid)
            reward = 1 if next_state == current_goal_pos else 0

            update_q_table(q_table, state, action, reward, next_state, alpha, gamma)

            # Model update
            model[(state, action)] = (reward, next_state)

            # Planning
            for _ in range(n_planning_steps):
                state_prime, action_prime = random.choice(list(model.keys()))
                reward_prime, next_state_prime = model[(state_prime, action_prime)]
                update_q_table(q_table, state_prime, action_prime, reward_prime, next_state_prime, alpha, gamma)

            state = next_state
            steps += 1
        if(len(steps_per_episode) == 0):
            steps_per_episode.append(steps)
        else:
            steps_per_episode.append(steps_per_episode[-1] + steps)

        if(flag_change == 0):
            if(steps == 16):
                min_steps_map1 = steps
                temp_table = copy.deepcopy(q_table)
                #best_path_map1, actions_map1 = find_shortest_path(temp_table, agent_pos1, goal_pos1, grid1)
        else:
            if(steps == 16):
                min_steps_map2 = steps
                temp_table = copy.deepcopy(q_table)
                #best_path_map2, actions_map2 = find_shortest_path(temp_table, agent_pos2, goal_pos2, grid2)

        if (steps_per_episode[-1] >= 5000 and flag_change == 0):
            current_agent_pos = agent_pos2
            current_goal_pos = goal_pos2
            current_grid = grid2
            flag_change = 1
            break
        elif (steps_per_episode[-1] >= 10000):
            break


    #print(f"Actions for optimal policy for first map with Dyna-Q: {actions_map1}")
    #print(f"Actions for optimal policy for second map with Dyna-Q: {actions_map2}")

    #print(min_steps_map1)
    #print(min_steps_map2)
    return steps_per_episode, q_table



def dyna_q_plus(grid, agent_pos, goal_pos, n_planning_steps, k=0.001, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    q_table = initialize_q_table(grid)
    model = {}
    steps_per_episode = []
    time_elapsed = {}

    for _ in range(episodes):
        state = agent_pos
        steps = 0

        while state != goal_pos:
            action = choose_action(state, q_table, epsilon)
            next_state = perform_action(state, action, grid)
            reward = 1 if next_state == goal_pos else 0

            # Dyna-Q+ Update
            update_q_table(q_table, state, action, reward, next_state, alpha, gamma)

            # Model update
            model[(state, action)] = (reward, next_state)
            time_elapsed[(state, action)] = 0
            for key in time_elapsed.keys():
                if (key == (state, action)):
                    continue
                else:
                    time_elapsed[key] += 1

            # Planning with Dyna-Q+
            for _ in range(n_planning_steps):
                state_prime, action_prime = random.choice(list(model.keys()))
                reward_prime, next_state_prime = model[(state_prime, action_prime)]
                # Modified reward for planning
                #updated_reward = reward_prime + (
                #            0.1 * (reward_prime - np.max(q_table[next_state_prime[1], next_state_prime[0]])))
                elapsed_time = time_elapsed[(state_prime, action_prime)]
                time_adjustment = k * np.sqrt(elapsed_time)
                updated_reward = reward_prime + time_adjustment
                update_q_table(q_table, state_prime, action_prime, updated_reward, next_state_prime, alpha, gamma)

            state = next_state
            steps += 1

        steps_per_episode.append(steps)

    return q_table, steps_per_episode

def dyna_q_plus_experiment(file_path1, file_path2, k, n_planning_steps=50, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):

    grid1, agent_pos1, goal_pos1 = parse_layout(file_path1)
    grid2, agent_pos2, goal_pos2 = parse_layout(file_path2)

    current_agent_pos = agent_pos1
    current_goal_pos = goal_pos1
    current_grid = grid1

    q_table = initialize_q_table(current_grid)
    model = {}
    steps_per_episode = []
    time_elapsed = {}
    flag_change = 0

    min_steps_map1 = np.inf
    min_steps_map2 = np.inf

    while True:
        state = current_agent_pos
        steps = 0


        while state != current_goal_pos:
            action = choose_action(state, q_table, epsilon)
            next_state = perform_action(state, action, current_grid)
            reward = 1 if next_state == current_goal_pos else 0

            # Dyna-Q+ Update
            update_q_table(q_table, state, action, reward, next_state, alpha, gamma)

            # Model update
            model[(state, action)] = (reward, next_state)
            time_elapsed[(state, action)] = 0
            for key in time_elapsed.keys():
                if (key == (state, action)):
                    continue
                else:
                    time_elapsed[key] += 1

            # Planning with Dyna-Q+
            for _ in range(n_planning_steps):
                state_prime, action_prime = random.choice(list(model.keys()))
                reward_prime, next_state_prime = model[(state_prime, action_prime)]
                # Modified reward for planning
                #updated_reward = reward_prime + (
                #            0.1 * (reward_prime - np.max(q_table[next_state_prime[1], next_state_prime[0]])))
                elapsed_time = time_elapsed[(state_prime, action_prime)]
                time_adjustment = k * np.sqrt(elapsed_time)
                updated_reward = reward_prime + time_adjustment
                update_q_table(q_table, state_prime, action_prime, updated_reward, next_state_prime, alpha, gamma)

            state = next_state
            steps += 1
        if (len(steps_per_episode) == 0):
            steps_per_episode.append(steps)
        else:
            steps_per_episode.append(steps_per_episode[-1] + steps)

        if (flag_change == 0):
            if (steps == 16):
                min_steps_map1 = steps
                temp_table = copy.deepcopy(q_table)

                #best_path_map1, actions_map1 = find_shortest_path(temp_table, agent_pos1, goal_pos1, grid1)

        else:
            if (steps == 10):
                min_steps_map2 = steps
                temp_table == copy.deepcopy(q_table)

                #best_path_map2, actions_map2 = find_shortest_path(temp_table, agent_pos2, goal_pos2, grid2)



        if (steps_per_episode[-1] >= 5000 and flag_change == 0):
            current_agent_pos = agent_pos2
            current_goal_pos = goal_pos2
            current_grid = grid2
            flag_change = 1
            break


        elif(steps_per_episode[-1] >= 10000):
            break


    #print(f"Actions for optimal policy for first map with Dyna-Q+: {actions_map1}")
    #print(f"Actions for optimal policy for second map with Dyna-Q+: {actions_map2}")

    #print(min_steps_map1)
    #print(min_steps_map2)
    return steps_per_episode, q_table



def dyna_q_prioritized_sweeping(grid, agent_pos, goal_pos, n_planning_steps, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1,):
    flag_first_time = 0
    q_table = initialize_q_table(grid)
    model = {}
    steps_per_episode = []
    priority_queue = []


    for episode in range(episodes):
        state = agent_pos
        steps = 0

        while state != goal_pos:
            action = choose_action(state, q_table, epsilon)
            next_state = perform_action(state, action, grid)
            reward = 1 if next_state == goal_pos else 0

            # Dyna-Q Update
            update_q_table(q_table, state, action, reward, next_state, alpha, gamma)

            # Model update
            model[(state, action)] = (reward, next_state)
            # Add to priority queue
            heapq.heappush(priority_queue,
                           (-abs(reward - np.max(q_table[next_state[1], next_state[0]])), state, action))

            # Planning with Prioritized Sweeping
            for _ in range(n_planning_steps):
                if not priority_queue:
                    break
                _, state_prime, action_prime = heapq.heappop(priority_queue)
                reward_prime, next_state_prime = model.get((state_prime, action_prime), (0, state_prime))
                update_q_table(q_table, state_prime, action_prime, reward_prime, next_state_prime, alpha, gamma)
                # Add neighbors to the priority queue
                for neighbor in get_neighbors(state_prime, grid):
                    heapq.heappush(priority_queue, (
                    -abs(reward_prime - np.max(q_table[neighbor[1], neighbor[0]])), neighbor, action_prime))

            state = next_state
            steps += 1

        steps_per_episode.append(steps)
        if (steps == 14):
            if (flag_first_time == 0):
                first_time_optimal_policy = episode
                print("First time reaching optimal policy: ", episode)
                flag_first_time = 1

    return q_table, steps_per_episode, first_time_optimal_policy
def get_neighbors(state, grid):
    x, y = state
    neighbors = []
    if x > 0: neighbors.append((x - 1, y))  # left
    if x < len(grid[0]) - 1: neighbors.append((x + 1, y))  # right
    if y > 0: neighbors.append((x, y - 1))  # up
    if y < len(grid) - 1: neighbors.append((x, y + 1))  # down
    return neighbors

"""
file_path = "maps/map.txt"
grid, agent_pos, goal_pos = parse_layout(file_path)

first_time_q = []
steps_q = []

first_time_dyna_q_5 = []
steps_dyna_q_5 = []

first_time_dyna_q_50 = []
steps_dyna_q_50 = []

first_time_dyna_q_5_pw = []
steps_dyna_q_5_pw = []

first_time_dyna_q_50_pw = []
steps_dyna_q_50_pw = []


for _ in range(100):
    q_table, steps_per_episode_q, first_time_optimal_policy_q = q_learning(grid, agent_pos, goal_pos)
    shortes_path_q, actions_q = find_shortest_path(q_table, agent_pos, goal_pos, grid)
    first_time_q.append(first_time_optimal_policy_q)
    steps_q.append(steps_per_episode_q)
    print("Q-learning finished")
    #print(shortes_path_q)
    #print(actions_q)



    dyna_q_table_5, steps_per_episode_dyna_q_5, first_time_optimal_policy_dyna_q_5 = dyna_q(grid, agent_pos, goal_pos, n_planning_steps=5)
    shortes_path_dyna_q, actions_q = find_shortest_path(dyna_q_table_5, agent_pos, goal_pos, grid)
    first_time_dyna_q_5.append(first_time_optimal_policy_dyna_q_5)
    steps_dyna_q_5.append(steps_per_episode_dyna_q_5)
    print("Dyna-Q (n_learning_steps=5) learning finished")
    #print(shortes_path_q)
    #print(actions_q)


    dyna_q_table_50, steps_per_episode_dyna_q_50, first_time_optimal_policy_dyna_q_50 = dyna_q(grid, agent_pos, goal_pos, n_planning_steps=50)
    shortes_path_dyna_q, actions_q = find_shortest_path(dyna_q_table_50, agent_pos, goal_pos, grid)
    first_time_dyna_q_50.append(first_time_optimal_policy_dyna_q_50)
    steps_dyna_q_50.append(steps_per_episode_dyna_q_50)
    print("Dyna-Q (n_learning_steps=50) learning finished")
    #print(shortes_path_q)
    #print(actions_q)


    dyna_q_table_5_pw, steps_per_episode_dyna_q_5_pw, first_time_optimal_policy_dyna_q_5_pw = dyna_q(grid, agent_pos, goal_pos, n_planning_steps=5)
    first_time_dyna_q_5_pw.append(first_time_optimal_policy_dyna_q_5_pw)
    steps_dyna_q_5_pw.append(steps_per_episode_dyna_q_5_pw)

    dyna_q_table_50_pw, steps_per_episode_dyna_q_50_pw, first_time_optimal_policy_dyna_q_50_pw = dyna_q(grid, agent_pos, goal_pos, n_planning_steps=50)
    first_time_dyna_q_50_pw.append(first_time_optimal_policy_dyna_q_50_pw)
    steps_dyna_q_50_pw.append(steps_per_episode_dyna_q_50_pw)



first_time_q = np.array(first_time_q)
first_time_dyna_q_5 = np.array(first_time_dyna_q_5)
first_time_dyna_q_50 = np.array(first_time_dyna_q_50)
first_time_dyna_q_5_pw = np.array(first_time_dyna_q_5_pw)
first_time_dyna_q_50_pw = np.array(first_time_dyna_q_50_pw)


mean_optimal_policy_q = np.mean(first_time_q)
mean_optimal_policy_dyna_q_5 = np.mean(first_time_dyna_q_5)
mean_optimal_policy_dyna_q_50 = np.mean(first_time_dyna_q_50)
mean_optimal_policy_dyna_q_5_pw = np.mean(first_time_dyna_q_5_pw)
mean_optimal_policy_dyna_q_50_pw = np.mean(first_time_dyna_q_50_pw)

print(f"Mean of episodes for Q-learning to reach optimal policy: {mean_optimal_policy_q}")
print(f"Mean of episodes for Dyna-Q (n_learning_steps=5) to reach optimal policy: {mean_optimal_policy_dyna_q_5}")
print(f"Mean of episodes for Dyna-Q (n_learning_steps=50) to reach optimal policy: {mean_optimal_policy_dyna_q_50}")
print(f"Mean of episodes for Dyna-Q (n_learning_steps=5) with prioritize sweeping to reach optimal policy: {mean_optimal_policy_dyna_q_5_pw}")
print(f"Mean of episodes for Dyna-Q (n_learning_steps=50) with prioritize sweeping to reach optimal policy: {mean_optimal_policy_dyna_q_50_pw}")

steps_q = np.array(steps_q)
steps_dyna_q_5 = np.array(steps_dyna_q_5)
steps_dyna_q_50 = np.array(steps_dyna_q_50)
steps_dyna_q_5_pw = np.array(steps_dyna_q_5_pw)
steps_dyna_q_50_pw = np.array(steps_dyna_q_50_pw)

avg_steps_q = np.mean(steps_q, axis=0)
avg_steps_dyna_q_5 = np.mean(steps_dyna_q_5, axis=0)
avg_steps_dyna_q_50 = np.mean(steps_dyna_q_50, axis=0)
avg_steps_dyna_q_5_pw = np.mean(steps_dyna_q_5_pw, axis=0)
avg_steps_dyna_q_50_pw = np.mean(steps_dyna_q_50_pw, axis=0)

plt.figure()
plt.plot(list(range(1, 301)), avg_steps_q[:300], label="Broj koraka tokom planiranja = 0")
plt.plot(list(range(1, 301)), avg_steps_dyna_q_5[:300], label="Broj koraka tokom planiranja = 5")
plt.plot(list(range(1, 301)), avg_steps_dyna_q_50[:300], label="Broj koraka tokom planiranja = 50")
plt.xlabel("Epizode")
plt.ylabel("Broj koraka po epizodi")
plt.title("Grafik zavisnosti broja koraka po epizodi od epizode")
plt.legend()
plt.ylim(0, 200)
plt.grid(True)
plt.show()


plt.figure()
plt.plot(list(range(1, 101)), avg_steps_dyna_q_5[:100], label="Bez prioritetnog odredjivanja stanja")
plt.plot(list(range(1, 101)), avg_steps_dyna_q_5_pw[:100], label="Sa prioritetnim odredjivanjem stanja")
plt.xlabel("Epizode")
plt.ylabel("Broj koraka po epizodi")
plt.title("Grafik zavisnosti broja koraka po epizodi od epizode (broj koraka tokom planiranja = 5)")
plt.legend()
plt.ylim(0, 100)
plt.grid(True)
plt.show()

plt.figure()
plt.plot(list(range(1, 101)), avg_steps_dyna_q_50[:100], label="Bez prioritetnog odredjivanja stanja")
plt.plot(list(range(1, 101)), avg_steps_dyna_q_50_pw[:100], label="Sa prioritetnim odredjivanjem stanja")
plt.xlabel("Epizode")
plt.ylabel("Broj koraka po epizodi")
plt.title("Grafik zavisnosti broja koraka po epizodi od epizode (broj koraka tokom planiranja = 50)")
plt.legend()
plt.ylim(0, 100)
plt.grid(True)
plt.show()
"""


def pad_sublists(data):
    # Step 1: Determine the maximum length of sublists
    max_length = max(len(sublist) for sublist in data)

    # Step 2: Pad each sublist to the maximum length with None
    padded_data = [sublist + [None] * (max_length - len(sublist)) for sublist in data]

    return padded_data

def mean_ignore_none(column):
    # Filter out None values and compute the mean of the remaining elements
    filtered = [x for x in column if x is not None]
    if filtered:  # Ensure there are elements to avoid division by zero
        return sum(filtered) / len(filtered)
    return None  # Return None if the column is empty or contains only None

def mean_along_columns(padded_data, pad_value=None):
    # Convert the list of lists into a NumPy array
    np_array = np.array(padded_data, dtype=object)
    # Compute the mean along columns, ignoring pad_value
    means = [mean_ignore_none(np_array[:, col]) for col in range(np_array.shape[1])]
    return means
def experiment():

    file_path1 = "maps/map2.txt"
    file_path2 = ("maps/map3.txt")

    steps_dyna_q = []
    steps_dyna_q_plus = []
    steps_dyna_q_plus_1 = []
    steps_dyna_q_plus_2 = []



    for i in range(100):
        steps_per_episode_dyna_q, dyna_q_table = dyna_q_experiment(file_path1, file_path2)
        steps_per_episode_dyna_q.insert(0, 0)

        steps_per_episode_dyna_q_plus, dyna_q_plus_table = dyna_q_plus_experiment(file_path1, file_path2, k=0.1)
        steps_per_episode_dyna_q_plus.insert(0, 0)

        steps_per_episode_dyna_q_plus_1, dyna_q_plus_table_1 = dyna_q_plus_experiment(file_path1, file_path2, k=0.01)
        steps_per_episode_dyna_q_plus_1.insert(0, 0)

        steps_per_episode_dyna_q_plus_2, dyna_q_plus_table_2 = dyna_q_plus_experiment(file_path1, file_path2, k=0.001)
        steps_per_episode_dyna_q_plus_2.insert(0, 0)

        steps_dyna_q.append(steps_per_episode_dyna_q)
        steps_dyna_q_plus.append(steps_per_episode_dyna_q_plus)
        steps_dyna_q_plus_1.append(steps_per_episode_dyna_q_plus_1)
        steps_dyna_q_plus_2.append(steps_per_episode_dyna_q_plus_2)


    steps_dyna_q = pad_sublists(steps_dyna_q)
    steps_dyna_q_plus = pad_sublists(steps_dyna_q_plus)
    steps_dyna_q_plus_1 = pad_sublists(steps_dyna_q_plus_1)
    steps_dyna_q_plus_2 = pad_sublists(steps_dyna_q_plus_2)


    #steps_dyna_q = np.array(steps_dyna_q)
    #steps_dyna_q_plus = np.array(steps_dyna_q_plus)


    avg_steps_dyna_q = mean_along_columns(steps_dyna_q)
    avg_steps_dyna_q_plus = mean_along_columns(steps_dyna_q_plus)
    avg_steps_dyna_q_plus_1 = mean_along_columns(steps_dyna_q_plus_1)
    avg_steps_dyna_q_plus_2 = mean_along_columns(steps_dyna_q_plus_2)

    rewards_dyna_q = list(range(1, len(avg_steps_dyna_q) + 1))
    rewards_dyna_q_plus = list(range(1, len(avg_steps_dyna_q_plus) + 1))
    rewards_dyna_q_plus_1 = list(range(1, len(avg_steps_dyna_q_plus_1) + 1))
    rewards_dyna_q_plus_2 = list(range(1, len(avg_steps_dyna_q_plus_2) + 1))


    plt.figure()
    plt.plot(avg_steps_dyna_q, rewards_dyna_q, linewidth=2, label=r'Dyna-Q')
    plt.plot(avg_steps_dyna_q_plus, rewards_dyna_q_plus, linewidth=2, label=r'Dyna-Q+($\kappa = {}$)'.format(0.1))
    plt.plot(avg_steps_dyna_q_plus_1, rewards_dyna_q_plus_1, linewidth=2, label=r'Dyna-Q+($\kappa = {}$)'.format(0.01))
    plt.plot(avg_steps_dyna_q_plus_2, rewards_dyna_q_plus_2, linewidth=2, label=r'Dyna-Q+($\kappa = {}$)'.format(0.001))
    plt.xlabel("Vremenski korak")
    plt.ylabel("Kumulativna nagrada")
    plt.title("Kumulativna nagrada u zavisnosti od vremenskog koraka")
    plt.legend()
    plt.grid(True)
    plt.show()

    return
experiment()





