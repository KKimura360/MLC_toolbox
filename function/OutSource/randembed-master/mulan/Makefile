.SECONDARY:

.PHONY: all rebuildmat

all: 
	@echo "*** if you lack a poissrnd function, try 'make poissrnd.m' ***" 1>&2
	@echo "*** otherwise, run traintestall.m from matlab ***" 1>&2
	@echo "*** and go get a sandwich ***" 1>&2

oissrnd.m:
	wget -q http://www.mathworks.com/matlabcentral/fileexchange/1261-poissrnd/content/oissrnd.m

poissrnd.m: oissrnd.m
	@perl -MHTML::Entities -e '$$l=join "",<>; $$l=~m%<pre class="matlab-code">(.*?)</pre>%s; $$s=$$1; $$s=~s/break/return/; print decode_entities($$s)' $< > $@

rebuildmat: bibtex.mat corel5k.mat mediamill.mat yeast.mat
	make mostlyclean

%.mat: %-train.x.txt %-train.y.txt %-test.x.txt %-test.y.txt
	matlab -nodisplay -nojvm -r "txtz2mat('$*'); exit;"

%-train.x.txt %-train.y.txt %-test.x.txt %-test.y.txt: %-train.csv %-test.csv %.xml
	python xmlcsv2xy.py $*

%.csv: %.arff
	java -cp /usr/share/java/weka.jar weka.core.converters.CSVSaver -i $< -o $@

%-train.arff %-test.arff %.xml: %.rar
	unrar x -y -cl -inul $< $*.xml $*-train.arff $*-test.arff
	touch $*-train.arff $*-test.arff $*.xml

%.rar:
	wget http://sourceforge.net/projects/mulan/files/datasets/$@

mostlyclean:
	rm -f *.csv *.arff *.bak *.xml *.txt

clean: mostlyclean
	rm -f *.mat

.PHONY: clean mostlyclean
