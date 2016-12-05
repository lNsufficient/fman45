function [state_action_feats, prev_grid, prev_head_loc] = extract_state_action_features(prev_grid, grid, prev_head_loc, nbr_feats)
%
% Code may be changed in this function, but only where it states that it is 
% allowed to do so
%
% Code part of ML-2016
%
% Function to extract state-action features, based on current and previous
% grids (game screens)
%
% Input:
%
% prev_grid     - Previous grid (game screen), N-by-N matrix. If initial
%                 time-step: prev_grid = grid, else: prev_grid != grid.
% grid          - Current grid (game screen), N-by-N matrix. If initial
%                 time-step: prev_grid = grid, else: prev_grid != grid.
% prev_head_loc - The previous location of the head of the snake (from the 
%                 previous time-step). If initial time-step: Assumed known,
%                 else: inferred in function "update_snake_grid.m" (so in
%                 practice it will always be known in this function)
% nbr_feats     - Number of state-action features per action. Set this 
%                 value appropriately in the calling script "snake.m", to
%                 match the number of state-action features per action you
%                 end up using
%
% Output:
%
% state_action_feats - nbr_feats-by-|A| matrix, where |A| = number of
%                      possible actions (|A| = 3 in Snake), and nbr_feats
%                      is described under "Input" above. This matrix
%                      represents the state-action features extracted given
%                      the current and previous grids (game screens)
% prev_grid          - The previous grid as seen from one step in the
%                      future, i.e., prev_grid is set to the input grid
% prev_head_loc      - The previous head location as seen from one step
%                      in the future, i.e., prev_head_loc is set to the
%                      current head location (the current head location is
%                      inferred in the code below)
%
% Bugs, ideas etcetera: send them to the course email

% Extract grid size
N = size(grid, 1);

% Initialize state_action_feats to nbr_feats-by-3 matrix
state_action_feats = nan(nbr_feats, 3);

% Based on how grid looks now and at previous time step, infer head location
change_grid = grid - prev_grid;
prev_grid   = grid; % Used in later calls to "extract_state_action_features.m"

% Find head location (initially known that it is in center of grid)
if nnz(change_grid) > 0 % True, except in initial time-step
    [head_loc_m, head_loc_n] = find(change_grid > 0);
else % True only in initial time-step
    head_loc_m = round(N / 2);
    head_loc_n = round(N / 2);
end
head_loc = [head_loc_m, head_loc_n];

% Previous head location
prev_head_loc_m = prev_head_loc(1);
prev_head_loc_n = prev_head_loc(2);

% Infer current movement directory (N/E/S/W) by looking at how current and previous
% head locations are related
if prev_head_loc_m == head_loc_m + 1 && prev_head_loc_n == head_loc_n     % NORTH
    movement_dir = 1;
elseif prev_head_loc_m == head_loc_m && prev_head_loc_n == head_loc_n - 1 % EAST
    movement_dir = 2;
elseif prev_head_loc_m == head_loc_m - 1 && prev_head_loc_n == head_loc_n % SOUTH
    movement_dir = 3;
else                                                                      % WEST
    movement_dir = 4;
end

% The current head_loc will at the next time-step be prev_head_loc
prev_head_loc = head_loc;

% HERE BEGINS YOUR STATE-ACTION FEATURE ENGINEERING. ALL CODE BELOW IS 
% ALLOWED TO BE CHANGED IN ACCORDANCE WITH YOUR CHOSEN FEATURES. 
% Some skeleton code is provided to help you get started. Also, have a 
% look at the function "get_next_info" (see bottom of this function).
% You may find it useful.

[apple_loc_m, apple_loc_n] = find(grid == -1);
apple_loc = [apple_loc_m, apple_loc_n];

feature_index = 1;
for action = 1 : 3 % Evaluate all the different actions (left, forward, right)
    
    % Feel free to uncomment below line of code if you find it useful
    [next_head_loc, next_move_dir] = get_next_info(action, movement_dir, head_loc);
    
    % Replace this to fit the number of state-action features per features
    % you choose (3 are used below), and of course replace the randn() 
    % by something more sensible
    current_apple_vector = head_loc - apple_loc;
    next_apple_vector = next_head_loc - apple_loc;
    
    current_1_norm_apple_dist = sum(abs(current_apple_vector));
    next_1_norm_apple_dist = sum(abs(next_apple_vector));
    
%     current_2_norm = sum(current_apple_vector.^2);
%     next_2_norm = sum(next_apple_vector.^2);
%     
%     current_inf_norm = max(abs(current_apple_vector));
%     next_inf_norm = max(abs(next_apple_vector));
    %state_action_feats(feature_index, action) = current_1_norm_apple_dist - next_1_norm_apple_dist;
    
    state_action_feats(feature_index, action) = next_1_norm_apple_dist-current_1_norm_apple_dist;
    feature_index = feature_index + 1;
    
    
    
%    state_action_feats(feature_index, action) = current_2_norm - next_2_norm;
 %   feature_index = feature_index + 1;
    
 %   state_action_feats(feature_index, action) = current_inf_norm - next_inf_norm;
 %   feature_index = feature_index + 1;
    
    %Change in number of number of 4-neighbour bwlabels
    [current_l, current_NUM] = bwlabel(grid~=1, 4);
    
    h = hist(current_l(:), 0:current_NUM);
    h = h(2:end);
    [~,best_island] = max(h);
    is_good_island = current_l(next_head_loc(1), next_head_loc(2))==best_island;
  
    %state_action_feats(feature_index, action) = is_good_island*2-1;
    %feature_index = feature_index+1;  
    
    %     if ~is_good_island
%         disp('for testing purposes')
%     end
%         
        
    %Enters small area?
    next_grid = grid;
    %don't care about the last part of the tail being left, sometimes eats
    %apple
    next_grid(next_head_loc(1), next_head_loc(2)) = 1;
    [space_l, space_NUM] = bwlabel(grid ~= 1,4);
    
    island_sizes = hist(space_l(:), 0:space_NUM);
    [~, largest_island] = max(island_sizes(2:end));
    entering_island = space_l(next_head_loc(1), next_head_loc(2));
    is_ok = (entering_island == 0) + (entering_island==largest_island);
    island_badness = (is_ok==0)/island_sizes(entering_island+1)*sum(island_sizes);
    %island_size = island_sizes(entering_island+1);
    %island_goodness = island_size/sum(island_sizes(2:end))*(2*(entering_island>0)-1);
    
    
      
    state_action_feats(feature_index, action) = island_badness;
    feature_index = feature_index+1;
    
    %Check if will go inside smallest bwlabel
    
    %Deadly move?
    grid_value_next_head_loc = grid(next_head_loc(1), next_head_loc(2));
    state_action_feats(feature_index, action) = -grid_value_next_head_loc;
    feature_index = feature_index + 1;
    
    %Leftover single cell?
%     nbr_leftovers = grid(next_head_loc(1), next_head_loc(2)) == 0;
%     following_loc = next_head_loc + (head_loc - next_head_loc);
%     nbr_leftovers = nbr_leftovers + grid(following_loc(1), following_loc(2))==0;
%     nbr_leftovers = nbr_leftovers == 2;
%     state_action_feats(feature_index, action) = nbr_leftovers;
%     feature_index = feature_index + 1;

%    state_action_feats(3, action) = randn();
    feature_index = 1;
end
% if any(state_action_feats(2,:) == -1)
%     state_action_feats
% end
% if any(isnan(state_action_feats))
%     t=0;
% end
%state_action_feats
% if any(state_action_feats(2,:))
%     state_action_feats(2,:)
% end
end

function [next_head_loc, next_move_dir] = get_next_info(action, movement_dir, head_loc)
% Function to infer next haed location and movement direction

% Extract relevant stuff
head_loc_m = head_loc(1);
head_loc_n = head_loc(2);

if movement_dir == 1 % NORTH
    if action == 1     % left
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n - 1;
        next_move_dir   = 4; 
    elseif action == 2 % forward
        next_head_loc_m = head_loc_m - 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 1;
    else               % right
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n + 1;
        next_move_dir   = 2;
    end
elseif movement_dir == 2 % EAST
    if action == 1
        next_head_loc_m = head_loc_m - 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 1;
    elseif action == 2
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n + 1;
        next_move_dir   = 2;
    else
        next_head_loc_m = head_loc_m + 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 3;
    end
elseif movement_dir == 3 % SOUTH
    if action == 1
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n + 1;
        next_move_dir   = 2;
    elseif action == 2
        next_head_loc_m = head_loc_m + 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 3;
    else
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n - 1;
        next_move_dir   = 4;
    end
else % WEST
    if action == 1
        next_head_loc_m = head_loc_m + 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 3;
    elseif action == 2
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n - 1;
        next_move_dir   = 4;
    else
        next_head_loc_m = head_loc_m - 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 1;
    end
end
next_head_loc = [next_head_loc_m, next_head_loc_n];
end