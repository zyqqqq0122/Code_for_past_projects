function [entropy] = entropy_fwt(im,scale,psv,step_size)
    
    wave_coeff=fwt(im,scale,psv);

    N=length(wave_coeff);
    A=wave_coeff(1:N/2,1:N/2);
    V=wave_coeff(1+N/2:N,1:N/2);
    H=wave_coeff(1:N/2,1+N/2:N);
    D=wave_coeff(1+N/2:N,1+N/2:N);

    for k=1:length(step_size)
        
        delta=step_size(k);
        
        A_q(:,:,k)=quantizer(A,delta);
        V_q(:,:,k)=quantizer(V,delta);
        H_q(:,:,k)=quantizer(H,delta);
        D_q(:,:,k)=quantizer(D,delta);
        
        M=length(A_q(:,:,k));
        
        A1{k}=reshape(A_q(:,:,k),[1,M*M]);
        V1{k}=reshape(V_q(:,:,k),[1,M*M]);
        H1{k}=reshape(H_q(:,:,k),[1,M*M]);
        D1{k}=reshape(D_q(:,:,k),[1,M*M]);
        
        bins_A{k}=[min(A1{k}):delta:max(A1{k})];
        bins_V{k}=[min(V1{k}):delta:max(V1{k})];
        bins_H{k}=[min(H1{k}):delta:max(H1{k})];
        bins_D{k}=[min(D1{k}):delta:max(D1{k})];
        
        pdf_A{k}=hist(A1{k},bins_A{k})/length(A1{k});
        pdf_V{k}=hist(V1{k},bins_V{k})/length(V1{k});
        pdf_H{k}=hist(H1{k},bins_H{k})/length(H1{k});
        pdf_D{k}=hist(D1{k},bins_D{k})/length(D1{k});
        
        entro_A{k}=-sum(pdf_A{k}.*log2(pdf_A{k}+eps));
        entro_V{k}=-sum(pdf_V{k}.*log2(pdf_V{k}+eps));
        entro_H{k}=-sum(pdf_H{k}.*log2(pdf_H{k}+eps));
        entro_D{k}=-sum(pdf_D{k}.*log2(pdf_D{k}+eps));
        
        entro{k}=(entro_A{k}+entro_V{k}+entro_H{k}+entro_D{k})/4;

    end
    
    entropy=cell2mat(entro);
    
end

