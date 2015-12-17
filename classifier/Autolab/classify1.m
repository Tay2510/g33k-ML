function [Y] = classify1(Model1, X)

XTrain=X;
cellSize = 8;
XTrainFeat = [];
%w = waitbar(0,'Extracting HoG (Training Set)...');
size_Xtrain = size(XTrain, 1);
for i = 1 : size_Xtrain
    im = im2single(reshape(XTrain(i,:), 32, 32, 3));
    hog = vl_hog(im, cellSize);
    XTrainFeat = [XTrainFeat;hog(:)'];
   % waitbar(i / size_Xtrain);
end
XTrainFeat=double(XTrainFeat);
[N,D]=size(XTrainFeat);


Y=zeros(N,1);
mid=497;
w1=Model1.w1;
w2=Model1.w2;
a=Model1.a;
b=Model1.b;
for i=1:1:N
    x=XTrainFeat(i,:);
    for j=1:1:mid
        x=double(x);
        temp(1,j)=x * w1(j,:)'+a(j);
        out(1,j)=1/(1+exp(-temp(1,j)));
        %out(1,j)=(2/(1+exp(-2*(temp(1,j)))))-1;
    end
    y= out*w2 +b';
   [v id]=max(y);
    Y(i)=id;
end
Y=Y-1;
end

