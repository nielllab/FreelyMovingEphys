function Y = typecastMulti(X, datatype)
% typecastMulti
% Typecasing for multidimensional arrays.
%
% Equivilent to typecast(), but this will work for multidimensional arrays.
%
% `X` is an array of any shape
% `datatype` is the datatype that `X` will be typecast to. The
% array returned, Y, will be cast to the new datatype without
% changing the underlying data.
%
% Written by DMM, June 2023
%
  sz = num2cell(size(X));

  Y = reshape(typecast(X(:), datatype), [], sz{2:end});

end

