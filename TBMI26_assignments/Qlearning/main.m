


level = 4;                     % Values 1,2,3,4 for different levels
nEpisodes = 4000;             % Amount of episodes
gamma = 0.999;%0.3;                   % Discount factor
initAlpha = 0.2;%0.9;               % Confidence
initEpsilon = 0.8;%0.9;             % Prob. of random
actions = [1, 2, 3, 4];        % All actions
probs = [1/4, 1/4, 1/4, 1/4];  % Distribution over actions

gwinit(level);
state = gwstate();
Q = randn(state.xsize, state.ysize, size(actions, 2));
Q(1,:,2)   = -inf;
Q(end,:,1) = -inf;
Q(:,1,4)   = -inf;
Q(:,end,3) = -inf;
tic
for n = 1:nEpisodes
    if ~mod(n, 100)
        toc
        tic
        n
    end
    
    %if (n > nEpisodes * 0.9)
    %    gwplotarrow([x,y],a);
    %    gwdraw();
    %end
        
    gwinit(level);
    state = gwstate();
    epsilon = initEpsilon*(exp(-n/(nEpisodes*0.5)));
    alpha = initAlpha;
    %alpha   = initAlpha*(exp(-n/(nEpisodes*0.5)));
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
            
            % Action was not valid
            %Q(x, y, a) = -inf; 
        end
        
        r = state.feedback;
        x_prime = state.pos(1);
        y_prime = state.pos(2);
        [~, oa] = chooseaction(Q, x_prime, y_prime, ...
                              actions, probs, epsilon);
        
        
        Q(x, y, a) = (1 - alpha)*Q(x, y, a) + ...
                        alpha*(r + gamma*Q(x_prime, y_prime, oa));
    end
end
toc
%%
%Q = randn(state.xsize, state.ysize, size(actions, 2));
gwdraw();
gwplotallarrows(Q,actions,probs);

%%
gwinit(3);
gwdraw();
gwplotallarrows(Q,actions,probs);