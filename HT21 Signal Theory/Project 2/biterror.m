function [dttd] = biterror(dttd,ernum)
p=randperm(numel(dttd),ernum);
for i=1:numel(p)
    pt=p(i);
    dttd(pt)=dttd(pt)*(-1);
end
end