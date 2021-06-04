function files = get_files(fdir, prefix, ext, append_pre)

a = dir([fdir '/' prefix '*.' ext]);

if exist('append_pre','var')
    for k = 1:length(a)
        files{k} = fullfile(fdir,[append_pre a(k).name]);
    end
else
    for k = 1:length(a)
        files{k} = fullfile(fdir,[a(k).name]);
    end
end


end