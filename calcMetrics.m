function [E, R] = calcMetrics(inFile, outFile)
if length(inFile) == length(outFile)
    for i = 1:length(inFile)
        in = imread(inFile{i}); in = im2double(in(:,:,1));
        out = imread(outFile{i}); out = im2double(out(:,:,1));
        n_test_images = 50;
        E(i) = 1/n_test_images * sum(sum(abs(in-out)));
        R(i) = corr2(in, out);
    end
end