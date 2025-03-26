pdflatex.exe  -interaction=nonstopmode doc.tex
pdfcrop.exe  doc.pdf
pdftocairo.exe  -png -r 30000 -transp doc-crop.pdf