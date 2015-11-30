function [ Model ] = train_NN( data,labels )
XTrain=data;
cellSize = 16;
XTrainFeat = [];
w = waitbar(0,'Extracting HoG (Training Set)...');
size_Xtrain = size(XTrain, 1);
for i = 1 : size_Xtrain
    im = im2single(reshape(XTrain(i,:), 32, 32, 3));
    hog = vl_hog(im, cellSize);
    XTrainFeat = [XTrainFeat;hog(:)'];
    waitbar(i / size_Xtrain);
end
Y=labels;
[N,D]=size(XTrainFeat);
%XX=zeros(N,D/3);

%for i=1:1:N
%    for j=1:3:D-3
 %       XX(i,(j+2)/3)=X(i,j)*0.299+X(i,j+1)*0.587+X(i,j+2)*0.114;
  %  end
%end
        

in_unit=D;
mid_unit=590;
out_unit=1;

a=rands(mid_unit,1);
b=rands(out_unit,1);
w1=rands(mid_unit,in_unit);
w2=rands(mid_unit,out_unit);
%mid_t=rands(1,mid_unit);
%mid_out=rands(1,mid_unit);
w = waitbar(0,'Neural Network Training ...');
for niter=1:1:1
for i=1:1:N
    
    %select training data
    x=XTrainFeat(i,:);   
    %value of hiden layer unit
    for j=1:1:mid_unit
        x=double(x);
        mid_t(1,j)=x* w1(j,:)'+a(j);
        mid_out(1,j)=1/(1+exp(-mid_t(1,j)));
    end
    %output 
    y= mid_out*w2 +b;
    %calculate difference
    e=Y(i,1)-y;
    %adjustment
    
    for j=1:1:mid_unit
        temp=1/(1+exp(-mid_t(j)));
        f(j)=temp*(1-temp);
    end
    for k=1:1:in_unit
        for j=1:1:mid_unit
            adw1(k,j)=f(j)*x(k)*e*w2(j,1);
            ada(j,1)=f(j)*w2(j,1);
        end
    end
    adw1=double(adw1);
    ada=double(ada);
    %w2=double(w2);
    e=double(e);
    mid_out=double(mid_out);
    adw2=e*mid_out';
    adb=e;
    w1=w1+0.1*adw1';
    a=a+0.1*ada;
    b=b+0.1*adb;
    w2=w2+0.1*adw2;
    waitbar(i / N);

end
end
   Model.w1=w1;
   Model.w2=w2;
   Model.a=a;
   Model.b=b;
end

