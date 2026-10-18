# Training data

`second_order_ode.m` generates the arm trajectories used to train the model. It simulates 500 movements of 20 time steps each and writes them to an Excel file in `data/`. Edit the commented blocks to choose the movement directions (one, two, four or eight).

The training scripts read `data/train_random_theta_vel_dist.xlsx`. Generate it with this script before training.
