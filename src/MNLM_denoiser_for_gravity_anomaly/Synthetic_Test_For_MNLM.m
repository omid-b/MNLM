clc;
clear all;
close all;
%% Author: Hanbing Ai
%% How to contact me?
%E-mail:AHB_ECUT@163.com/1724178612@qq.com
%% Date:2022.11.13.
% Read the raw data
DATA_1=load('synthetic.xyz');DATA_2=load('10precent.xyz');
%
[D_1,X,Y]=transform_in(DATA_1,121,121);[D_2,XX,YY]=transform_in(DATA_2,121,121);
%%
%
INPUT=D_2; % Noise-corrupted data
TRUE=D_1; % Noise-free data
%
% figure
% %
% set(gcf,'color',[1 1 1],'units','normalized','position',[0 0 0.4 0.55])
% % 
% imagesc(TRUE-INPUT)
% colormap('hsv')
% colorbar
% set(gca,'FontSize',12)
%% Main program for MNLM to denoise
ds=5;% block size for calculate weight
Ds=[2:2:18];% search block
sigma=0.0001*mean(mean(INPUT));
h=[100:100:1000].*sigma;
%
%%
for i=1:length(Ds)
    for j=1:length(h)
       N_processed(i,j)= sqrt(mean(mean((D_1-NLM_II(INPUT,Ds(i),ds,h(j))).^2)));
    end
end
%
[R,C]=find(N_processed==min(min(N_processed)));
%
OUTPUT_MNLM=NLM_II(INPUT,Ds(R(1)),ds,h(C(1)));
Noise_component_MNLM=INPUT-OUTPUT_MNLM;
Difference_MNLM=D_1-OUTPUT_MNLM;
%
RMS_1=min(min(N_processed));
%
figure
%
set(gcf,'color',[1 1 1],'units','normalized','position',[0 0 0.6 0.7])
% 
subplot(2,2,1)
contourf(X,Y,Noise_component_MNLM)
colormap('jet')
colorbar
title('Noise component','FontSize',12)
set(gca,'FontSize',12)
subplot(2,2,2)
contourf(X,Y,OUTPUT_MNLM)
colorbar
title('Denoised result','FontSize',12)
set(gca,'FontSize',12)
subplot(2,2,3)
contourf(X,Y,Difference_MNLM)
colorbar
title('The difference between the denoised result and the noise-free data','FontSize',12)
set(gca,'FontSize',12)
subplot(2,2,4)
contourf(X,Y,INPUT)
colorbar
title('Noise corrupted result','FontSize',12)
set(gca,'FontSize',12)
% 
% figure
% %
% set(gcf,'color',[1 1 1],'units','normalized','position',[0 0 0.5 0.5])
% % 
% imagesc(Ds,h,N_processed)
% shading interp
% h_c=colorbar('southoutside');
% hold on
% % scatter(Ds(R(1)),h(C(1)),40,'markerfacecolor','r','markeredgecolor','k')
% colormap('bone')
% axis tight
% set(gca,'FontSize',14)
% set(gca,'Ydir','normal')
% h_x=xlabel('Ds');      
% h_y=ylabel('h');   
% set(h_x,'fontsize',14);
% set(h_y,'fontsize',14);
% set(gcf,'unit', 'normalized', 'position',[0.1 0.1 0.5 0.5])
% set(gcf,'color',[1 1 1])
%% SVD
[u,s,v]=svd(D_1);
%
s1=s;s0=s;
s0(abs(s1)<580)=0;
%
OUTPUT_SVD=u*s0*v';
Noise_component_SVD=INPUT-OUTPUT_SVD;
Difference_SVD=D_1-OUTPUT_SVD;
% 
figure
%
set(gcf,'color',[1 1 1],'units','normalized','position',[0 0 0.6 0.7])
% 
subplot(2,2,1)
contourf(X,Y,Noise_component_SVD)
colormap('jet')
colorbar
title('Noise component','FontSize',12)
set(gca,'FontSize',12)
subplot(2,2,2)
contourf(X,Y,OUTPUT_SVD)
colorbar
title('Denoised result','FontSize',12)
set(gca,'FontSize',12)
subplot(2,2,3)
contourf(X,Y,Difference_SVD)
colorbar
title('The difference between the denoised result and the noise-free data','FontSize',12)
set(gca,'FontSize',12)
% 
%% Wavelet
[c,s]=wavedec2(D_1,3,'sym4');  
OUTPUT_wavelet=wrcoef2('a',c,s,'sym4');
Noise_component_wavelet=INPUT-OUTPUT_wavelet;
Difference_wavelet=D_1-OUTPUT_wavelet;
% 
figure
%
set(gcf,'color',[1 1 1],'units','normalized','position',[0 0 0.6 0.7])
% 
subplot(2,2,1)
contourf(X,Y,Noise_component_wavelet)
colormap('jet')
colorbar
title('Noise component','FontSize',12)
set(gca,'FontSize',12)
subplot(2,2,2)
contourf(X,Y,OUTPUT_wavelet)
colorbar
title('Denoised result','FontSize',12)
set(gca,'FontSize',12)
subplot(2,2,3)
contourf(X,Y,Difference_wavelet)
colorbar
title('The difference between the denoised result and the noise-free data','FontSize',12)
set(gca,'FontSize',12)
% 
%% DCT
GRAVITY=dct2(D_1);
[m n]=size(GRAVITY);
%
QC1=zeros(m,n);
for i=1:10
for j=1:10
    QC1(i,j)=GRAVITY(i,j); 
end
end
%
OUTPUT_DCT=idct2(QC1);
Noise_component_DCT=INPUT-OUTPUT_DCT;
Difference_DCT=D_1-OUTPUT_DCT;
% 
figure
%
set(gcf,'color',[1 1 1],'units','normalized','position',[0 0 0.6 0.7])
% 
subplot(2,2,1)
contourf(X,Y,Noise_component_DCT)
colormap('jet')
colorbar
title('Noise component','FontSize',12)
set(gca,'FontSize',12)
subplot(2,2,2)
contourf(X,Y,OUTPUT_DCT)
colorbar
title('Denoised result','FontSize',12)
set(gca,'FontSize',12)
subplot(2,2,3)
contourf(X,Y,Difference_DCT)
colorbar
title('The difference between the denoised result and the noise-free data','FontSize',12)
set(gca,'FontSize',12)
%%
DT_MNLM=transform_out(OUTPUT_MNLM,Y(1,:),X(:,1));DT_X=DT_MNLM(:,2);DT_Y=DT_MNLM(:,1);
DT_SVD=transform_out(OUTPUT_SVD,Y(1,:),X(:,1));
DT_wavelet=transform_out(OUTPUT_wavelet,Y(1,:),X(:,1));
DT_DCT=transform_out(OUTPUT_DCT,Y(1,:),X(:,1));
DT_MNLM(:,1)=DT_X;DT_MNLM(:,2)=DT_Y;
DT_SVD(:,1)=DT_X;DT_SVD(:,2)=DT_Y;
DT_wavelet(:,1)=DT_X;DT_wavelet(:,2)=DT_Y;
DT_DCT(:,1)=DT_X;DT_DCT(:,2)=DT_Y;
%%
DT_Noise_component_MNLM=transform_out(Noise_component_MNLM,Y(1,:),X(:,1));
DT_Noise_component_SVD=transform_out(Noise_component_SVD,Y(1,:),X(:,1));
DT_Noise_component_wavelet=transform_out(Noise_component_wavelet,Y(1,:),X(:,1));
DT_Noise_component_DCT=transform_out(Noise_component_DCT,Y(1,:),X(:,1));
DT_Noise_component_MNLM(:,1)=DT_X;DT_Noise_component_MNLM(:,2)=DT_Y;
DT_Noise_component_SVD(:,1)=DT_X;DT_Noise_component_SVD(:,2)=DT_Y;
DT_Noise_component_wavelet(:,1)=DT_X;DT_Noise_component_wavelet(:,2)=DT_Y;
DT_Noise_component_DCT(:,1)=DT_X;DT_Noise_component_DCT(:,2)=DT_Y;
%%
DT_Difference_MNLM=transform_out(Difference_MNLM,Y(1,:),X(:,1));
DT_Difference_SVD=transform_out(Difference_SVD,Y(1,:),X(:,1));
DT_Difference_wavelet=transform_out(Difference_wavelet,Y(1,:),X(:,1));
DT_Difference_DCT=transform_out(Difference_DCT,Y(1,:),X(:,1));
DT_Difference_MNLM(:,1)=DT_X;DT_Difference_MNLM(:,2)=DT_Y;
DT_Difference_SVD(:,1)=DT_X;DT_Difference_SVD(:,2)=DT_Y;
DT_Difference_wavelet(:,1)=DT_X;DT_Difference_wavelet(:,2)=DT_Y;
DT_Difference_DCT(:,1)=DT_X;DT_Difference_DCT(:,2)=DT_Y;
%%
save denoised_result_SyntheticCase_MNLM.txt -ascii DT_MNLM
save denoised_result_SyntheticCase_SVD.txt -ascii DT_SVD 
save denoised_result_SyntheticCase_wavelet.txt -ascii DT_wavelet
save denoised_result_SyntheticCase_DCT.txt -ascii DT_DCT 
% 
save Noise_component_SyntheticCase_MNLM.txt -ascii DT_Noise_component_MNLM
save Noise_component_SyntheticCase_SVD.txt -ascii DT_Noise_component_SVD 
save Noise_component_SyntheticCase_wavelet.txt -ascii DT_Noise_component_wavelet
save Noise_component_SyntheticCase_DCT.txt -ascii DT_Noise_component_DCT 
% 
save Difference_SyntheticCase_MNLM.txt -ascii DT_Difference_MNLM
save Difference_SyntheticCase_SVD.txt -ascii DT_Difference_SVD 
save Difference_SyntheticCase_wavelet.txt -ascii DT_Difference_wavelet
save Difference_SyntheticCase_DCT.txt -ascii DT_Difference_DCT 