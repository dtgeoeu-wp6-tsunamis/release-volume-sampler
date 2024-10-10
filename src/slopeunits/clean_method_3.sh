#!/bin/bash
#
############################################################################
#
# MODULE:	clean_method_3.sh for GRASS 7
# AUTHOR(S):    Ivan Marchesini, Massimiliano Alvioli
# PURPOSE:	To create a raster layer of slope units 
# COPYRIGHT: (C) 2004-2012 by the GRASS Development Team
#
#		This program is free software under the GNU General Public
#		License (>=v2). Read the file COPYING that comes with GRASS
#		for details.
#
#############################################################################
#
input_vect=$1
output_vect=$2
minarea=$3
#
eval `g.region -pg`
smarea=`echo "10*$nsres*$ewres"|bc -l`
#slungh=`echo "15*($nsres+$ewres)/2"|bc -l|cut -f 1 -d "."`
v.clean input=$input_vect output=slu_clean tool=rmarea thres=$smarea --o --q
v.db.addcolumn map=slu_clean columns="area integer, perimetro integer" --q
v.to.db map=slu_clean option=area columns=area --q
v.to.db map=slu_clean option=perimeter columns=perimetro --q
v.db.droprow input=slu_clean where='area=""' output=slu_area --o --q
#
lista=`v.db.select -c map=slu_area where="area<=$minarea"|cut -f 1 -d "|"`
buchi=`echo $lista | sed 's/ /,/g'` 
totalebuchi=`echo "$lista"|wc -w`
v.extract input=slu_area cats=${buchi} output=slu_buchi type=area --o --q
v.to.rast input=slu_buchi output=slu_buchi use=cat --o --q
lista=`v.db.select -c map=slu_area where="area>$minarea"|cut -f 1 -d "|"`
nobuchi=`echo $lista | sed 's/ /,/g'` 
v.to.rast input=slu_nobuchi output=slu_nobuchi use=cat --o --q
v.extract input=slu_area cats=${nobuchi} output=slu_nobuchi type=area --o --q
v.category input=slu_area output=slu_bordi layer=2 type=boundary option=add --o --q
v.db.addtable slu_bordi layer=2 col="left integer,right integer,lunghezza integer" --q
v.to.db slu_bordi option=sides columns=left,right layer=2 type=boundary --q
v.to.db slu_bordi option=length columns=lunghezza layer=2 type=boundary --q
v.to.rast input=slu_area output=slu_area use=cat --o --q
r.mapcalc "coseno = cos(aspect_slu)" --o --q
r.mapcalc "seno = sin(aspect_slu)" --o --q
# GRASS6.4
#r.statistics2 base=slu_area cover=coseno method=count output=count --o --q
#r.statistics2 base=slu_area cover=coseno method=sum output=sumcos --o --q
#r.statistics2 base=slu_area cover=seno method=sum output=sumsin --o --q
# GRASS 7.0
r.stats.zonal base=slu_area cover=coseno method=count output=count --o --q
r.stats.zonal base=slu_area cover=coseno method=sum output=cumcos --o --q
r.stats.zonal base=slu_area cover=seno method=sum output=sumsin --o --q
#
r.mapcalc "cos_medio = sumcos/count" --o --q
r.mapcalc "sin_medio = sumsin/count" --o --q 
r.to.vect input=cos_medio output=cos_medio type=area --o --q
r.to.vect input=sin_medio output=sin_medio type=area --o --q
v.overlay ainput=slu_area binput=cos_medio operator=and atype=area btype=area output=cos_medio_over --o --q
v.overlay ainput=slu_area binput=sin_medio operator=and atype=area btype=area output=sin_medio_over --o --q
#
pulire=`v.category input=slu_buchi option=print --q`
#
g.copy vect=slu_area,$output_vect --q --o
#
ico=1
for i in ${pulire}
do 
    lista1=`db.select -c sql="select b2.right from slu_bordi_2 b2 where b2.left=$i and b2.right<>-1"`
    lista2=`db.select -c sql="select b2.left from slu_bordi_2 b2 where b2.left<>-1 and b2.right=$i"`
    vicini=`echo $lista1 $lista2 | sort -u | sed 's/ /,/g'` 
    if [ ! -z "$vicini" ]
    then
	echo " --- --- -- buco numero $ico di $totalebuchi, cat: $i, vicini: $vicini"
	ico=`echo "$ico+1"|bc -l|cut -f 1 -d "."`
	echo "vicini: $vicini"
	v.extract input=$output_vect cats=${vicini} output=intorno type=area --o --q
	chk_intorno=`v.category input=intorno type=centroid option=print --q`
	if [ ! -z "$chk_intorno" ]
	then
	    # potrei voler cambiare questo perche' quando ci sono buchi contigui fa un po' di casino
	    v.overlay ainput=intorno binput=slu_nobuchi output=intorno_OK atype=area btype=area olayer=0,1,0 operator=and --o --q
	    #
	    cos_buco=`v.db.select -c map=cos_medio_over where="a_cat=$i" columns=b_value --q`
	    sin_buco=`v.db.select -c map=sin_medio_over where="a_cat=$i" columns=b_value --q`
	    echo "buco cat $i: cos=$cos_buco sin=$sin_buco"
	    massimo=-10000
	    jmax=0
	    loop=`v.category input=intorno_OK option=print --q`
	    for j in ${loop}
	    do
		cos_j=`v.db.select -c map=cos_medio_over where="a_cat=$j" columns=b_value --q`
		sin_j=`v.db.select -c map=sin_medio_over where="a_cat=$j" columns=b_value --q`
		dotpr=`echo "($cos_buco*$cos_j+$sin_buco*$sin_j)*10000"|bc -l|cut -f 1 -d "."`
		if [ "$dotpr" -ge "$massimo" -a "$dotpr" -gt "0" ]
		then
		    massimo=$dotpr
		    jmax=$j
		fi
		echo $i $j $cos_j $sin_j $dotpr $jmax
	    done
	    echo "massimo: $massimo per j=$jmax"
	    if [ "$jmax" -gt 0 ]
	    then
		lunghezza=`db.select -c sql="select b2.lunghezza from slu_bordi_2 b2 where (b2.left=$i and b2.right=$jmax) or (b2.left=$jmax and b2.right=$i)"`
		perimetro=`v.db.select -c map=slu_clean columns=perimetro where="cat=$i" --q`
		if [ "$lunghezza" -gt "0" -a "$perimetro" -gt "0" ]
		then
		    frazione=`echo "$lunghezza/$perimetro*10000"|bc -l|cut -f 1 -d "."`
		    if [ "$frazione" -gt "500" ]
		    then
			echo "lungh: $lunghezza; perim: $perimetro; fract: $frazione"
			v.extract input=$output_vect output=slu_i cats=$i,$jmax -d new=$jmax --o --q
			v.overlay ainput=$output_vect binput=slu_i atype=area btype=area operator=not olayer=0,1,0 output=slu_j --o --q
			v.overlay ainput=slu_i binput=slu_j atype=area btype=area operator=or output=slu_k olayer=1,0,0 --o --q    	
			v.db.addcolumn map=slu_k column="newcat integer" --o --q
			v.db.update map=slu_k layer=1 column=newcat qcolumn=a_cat where="a_cat is not null" --o --q
			v.db.update map=slu_k layer=1 column=newcat qcolumn=b_cat where="b_cat is not null" --o --q
			v.reclass input=slu_k output=$output_vect column=newcat --o --q
			v.db.addtable map=$output_vect --q
			v.db.addcolumn map=$output_vect columns="area integer" --q
			v.to.db map=$output_vect option=area columns=area --q
			g.remove type=vector name=slu_i,slu_j,slu_k -f --q
		    fi
		fi
	    fi
	fi # chk_category
    fi # vicini
    g.remove type=vector name==intorno,intorno_OK -f --q
done
#
