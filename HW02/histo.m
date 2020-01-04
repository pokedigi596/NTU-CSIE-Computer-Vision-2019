function histo(file)
    Table = readtable(file,'Format', '%d');
    A = table2array(Table);
    x = 0 : 255;
    bar(x, A);
    saveas(gcf,'Histogram.png')
end