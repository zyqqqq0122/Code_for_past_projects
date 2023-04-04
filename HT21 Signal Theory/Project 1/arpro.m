


v=0:0.0001:0.5;
R1=(1+0.25*0.25-0.5*cos(2*pi*v)).^(-1);
figure(1);
plot(v,R1);
grid on;
xlabel('nomalized frequency');
ylabel('power spectral density');
hold on;
R0=abs((1-0.25*exp(-1i*2*pi*v)).^(-1));
for j=1:5001
    R3(j)=R0(j)*R0(j)*R1(j);
end

plot(v,R3);



    
  

