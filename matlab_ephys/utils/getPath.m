function filePath = getPath(d, k)
% filePath
% Search for a file given a directory (d) and a key (k) where the key
% has "*" and "?" characters to indicate wildcard characters/character (a
% "?" is only a single character, while a single "*" can be many.
%
% This returns the full filepath. If no match is found for the key in that
% directory, this will return nothing.
%
% Written by DMM, July 2023
%

pathstruct = dir(d+k);
filePath = string(pathstruct.folder) + '/' + string(pathstruct.name);

if (filePath == "/")
    sprintf("No file matching the key %s in directory %s.", k, rpath)
    return
end

end

