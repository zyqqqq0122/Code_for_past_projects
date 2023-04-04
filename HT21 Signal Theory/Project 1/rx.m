
fun = @(v,c) (1+0.25*0.25-0.5*cos(2*pi*v)).^(-1).*abs((1-0.25*exp(-1i*2*pi*v)).^(-1)).*abs((1-0.25*exp(-1i*2*pi*v)).^(-1)).*exp(1i*2*pi*v*c);
c=0;
ry(1)=integral(@(v) fun(v,c),-0.5,0.5);
for c=1:9
ry(c+1)=integral(@(v) fun(v,c),-0.5,0.5);
end
c=0:1:9;
stem(c,ry);




































































