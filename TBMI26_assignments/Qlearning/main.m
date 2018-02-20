

nEpisodes = 100000;
goalReward = 2000;
actions = [1, 2, 3, 4];
Q = randn(state.xsize, state.ysize, size(actions, 2));

alpha = 0.9;   % Confidence
gamma = 0.2;   % Discount factor
initEpsilon = 0.9; % Prob. of random
probs = [1/4, 1/4, 1/4, 1/4];

for n = 1:nEpisodes
    if ~mod(n, 100)
        n
    end
    
    if (n > nEpisodes * 0.9)
        gwplotarrow([x,y],a);
        gwdraw();
    end
        
    gwinit(1);
    state = gwstate();
    epsilon = initEpsilon*(exp(-n/(nEpisodes*0.2)));
    while ~state.isterminal
        x = state.pos(1);
        y = state.pos(2);
    
        while (true)
            
            [a, ~] = chooseaction(Q, x, y, ...
                              actions, probs, epsilon);
            gwaction(a);
            state = gwstate();
            if state.isvalid
                break;
            end
            
            % state is invalid
            Q(state.pos(1), state.pos(2), :) = -inf * ones(size(actions));           
        end
        
        x_prime = state.pos(1);
        y_prime = state.pos(2);
        [~, oa] = chooseaction(Q, x_prime, y_prime, ...
                              actions, probs, epsilon);
        r = 0;
        if state.isterminal
            %Q(x_prime, y_prime, :) = goalReward*[1, 1, 1, 1];
            r = goalReward;
        end
        
        Q(x, y, a) = (1 - alpha)*Q(x, y, a) + ...
                        alpha*(r + gamma*Q(x_prime, y_prime, oa));
    end
end
