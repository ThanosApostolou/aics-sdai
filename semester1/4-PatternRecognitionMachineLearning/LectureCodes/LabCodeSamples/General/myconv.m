function y = myconv(x,h)
y = zeros(1,length(x));
for k = 1:1:length(x)
y(k) = x([1:1:k])*h([k:-1:1])';
end;
end

