function [ cM ] = calcConfusionMatrix( Lclass, Ltrue )
   c_n = size(Lclass,1);
   c = sparse(Lclass, 1:c_n, ones(1,c_n));
   t_n = size(Ltrue, 1);
   t = sparse(Ltrue, 1:t_n, ones(1,t_n));
   [~, cM] = confusion(t, c);
end

