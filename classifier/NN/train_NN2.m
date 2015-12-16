function [ Model ] = train_NN( data,labels )
XTrain=data;
cellSize = 8;
XTrainFeat = [];
w = waitbar(0,'Extracting HoG (Training Set)...');
size_Xtrain = size(XTrain, 1);
for i = 1 : size_Xtrain
    im = im2single(reshape(XTrain(i,:), 32, 32, 3));
    hog = vl_hog(im, cellSize);
    XTrainFeat = [XTrainFeat;hog(:)'];
    waitbar(i / size_Xtrain);
end
[l,ll]=size(labels);
labels=labels+1;
%Y=labels;
Y=zeros(l,10);
for i=1:1:l
    Y(i,labels(i,1))=1;
end
%[XTrainFeat,XXX]=mapminmax(XTrainFeat);
%[Y,YYY]=mapminmax(Y);

XTrainFeat=double(XTrainFeat);
[XTrainFeat] = normalize(XTrainFeat);
[N,D]=size(XTrainFeat);
%display(XTrainFeat);
%display(Y);
    
in_unit=D;
mid_unit=600;
out_unit=10;

a=rands(mid_unit,1);
b=[0.1;0.1;0.1;0.1;0.1;0.1;0.1;0.1;0.1;0.1];
w1=rands(mid_unit,in_unit);
w2=rands(mid_unit,out_unit);
%display(a);
display(b);
%display(w1);
%display(w2);
%mid_t=rands(1,mid_unit);
%mid_out=rands(1,mid_unit);
w = waitbar(0,'Neural Network Training ...');
for niter=1:1:1
for i=1:1:N
    
    %select training data
    x=XTrainFeat(i,:); 
    display(x);
    %value of hiden layer unit
    for j=1:1:mid_unit
        x=double(x);
        mid_t(1,j)=x* w1(j,:)'+a(j);
        %display(mid_t(1,j));
        tempp=(2/(1+exp(-2*0.05*(mid_t(1,j)))))-1;
        %tempp=1/(1+exp(-mid_t(1,j)));
        display(tempp);
        mid_out(1,j)=tempp;
    end
    %output 
    y= mid_out*w2 +b';
    %calculate difference
    e=Y(i,:)-y;
    %adjustment
    
    for j=1:1:mid_unit
        temp=(2/(1+exp(-2*0.05*(mid_t(1,j)))))-1;
        %temp=1/(1+exp(-mid_t(1,j)));
        f(j)=temp*(1-temp);
    end
    for k=1:1:in_unit
        for j=1:1:mid_unit
            adw1(k,j)=f(j)*x(k)*(e(1)*w2(j,1)+e(2)*w2(j,2)+e(3)*w2(j,3)+e(4)*w2(j,4)+ ...
               e(5)*w2(j,5)+e(6)*w2(j,6)+e(7)*w2(j,7)+e(8)*w2(j,8)+e(9)*w2(j,9)+e(10)*w2(j,10)) ;
           %adw1(k,j)=f(j)*x(k)*(e(1)*w2(j,1)) ;
            ada(j,1)=f(j)*(e(1)*w2(j,1)+e(2)*w2(j,2)+e(3)*w2(j,3)+e(4)*w2(j,4)+ ...
               e(5)*w2(j,5)+e(6)*w2(j,6)+e(7)*w2(j,7)+e(8)*w2(j,8)+e(9)*w2(j,9)+e(10)*w2(j,10));
           %    ada(j,1)=f(j)*e(1)*w2(j,1);
        end
    end
    adw1=double(adw1);
    ada=double(ada);
    %w2=double(w2);
    e=double(e);
    mid_out=double(mid_out);
    adw2=mid_out'*e;
    adb=e';
    w1=w1+0.0001*adw1';
    a=a+0.0001*ada;
    b=b+0.0001*adb;
    w2=w2+0.0001*adw2;
    waitbar(i / N);

end
end
   Model.w1=w1;
   Model.w2=w2;
   Model.a=a;
   Model.b=b;
end

