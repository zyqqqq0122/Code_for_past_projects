function A = dct_coeff(M)
    
    i=(0:M-1)'*ones(1,M);
    k=ones(M,1)*(0:M-1);

    a=sqrt(2/M)*ones(M,1);
    a(1)=sqrt(1/M);
    alpha=a*ones(1,M);

    A=alpha.*cos((2*k+1).*i*pi/(2*M));
    
end

