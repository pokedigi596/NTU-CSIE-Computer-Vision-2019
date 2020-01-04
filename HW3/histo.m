function histo(file)
    Table = readtable(file,'Format', '%d');
    A = table2array(Table);
    x = 0 : 255;
    bar(x, A);
    file = split(file, '.');
    saveas(gcf,string(file{1, 1}) + '(histogram).png');
end