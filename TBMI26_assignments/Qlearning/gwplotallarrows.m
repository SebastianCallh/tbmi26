function gwplotallarrows(Q, actions, probs)
% Plot all arrows using gwplotarrow and the Q matrix.
  
global GWXSIZE;
global GWYSIZE;
global GWTERM;

% Arrow directions
% Change this to select arrow directions from the Q matrix.
A = ones(GWXSIZE, GWYSIZE);

for x = 1:GWXSIZE
   for y = 1:GWYSIZE
        [~, oa] = chooseaction(Q, x, y, actions, probs, 0);
        A(x, y) = oa;    
   end
end

for x = 1:GWXSIZE
    for y = 1:GWYSIZE
        if ~GWTERM(x,y)
            gwplotarrow([x y], A(x, y));
        end
    end
end

drawnow;
hold on;

    



