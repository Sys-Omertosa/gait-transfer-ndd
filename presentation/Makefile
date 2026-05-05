# Build presentation PDF (run from presentation/)

PDF = main.pdf
TEX = main.tex

.PHONY: pdf clean

pdf: $(PDF)

$(PDF): $(TEX)
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX)
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX)

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.synctex.gz $(PDF)
