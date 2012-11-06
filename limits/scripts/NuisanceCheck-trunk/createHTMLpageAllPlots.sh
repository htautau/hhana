#! /bin/sh

# you need to run this script from a dir above the 'dirname' dir

destination=$1
tagname=`basename $destination`
text=""
TNsize=400
nx=2

cd ${destination}


  mkdir eps
  mv *.eps eps
  mkdir png
  mv *.png png

  file="index.html"
  blue="#0000FF"
  red="#FF0000"
  black="#000000"
  title="${tagname} - ${dir}"

  rm -rf ${file}

  echo "<HTML>" >> ${file}
  echo "<TITLE>${title}</TITLE>" >> ${file}
  echo " <BODY>" >> ${file}
  echo "  <CENTER><H1><font color=\"${blue}\">${title}</font></H1></CENTER>" >> ${file}
  echo "<p>" >> ${file}
  echo ${text} >> ${file}
  echo "</p>" >> ${file}

  echo "  <TABLE>" >> ${file}
  echo "   <TR>" >> ${file}

  for t in $( "ls" * | grep -e ".txt" ); do
     echo "<a href=\"${t}\"><H3><font color=\"${blue}\"> ${t} </font></H3></a>" >> ${file}
  done

  #########################
  i1=0
  j1=0
  echo "    </TR>" >> ${file}
  echo "  </TABLE>" >> ${file}
  echo "<H1><font color=\"${blue}\">  </font></H1>" >> ${file}
  echo "  <TABLE>" >> ${file}
  echo "   <TR>" >> ${file}

  for epsfile1 in `cd eps; ls *.eps` ; do 

      eps1=$epsfile1
      img1=`echo ${eps1} | sed 's/.eps//'`

      #creating scaled thumbnails !!!
      png1=`echo ${eps1} | sed 's/.eps/.png/'`

      echo "    <td align=\"center\">" >> ${file}
      echo "      <a href=\"png/${png1}\"> <img src=\"png/${png1}\"> </a>" >> ${file}
      echo "      ${img1} <br><a href=\"png/${png1}\">[png]</a> <a href=\"eps/${eps1}\">[eps]</a>  "  >> ${file}
      echo "    </td>  "  >> ${file}
      i1=`expr $i1 + 1` 
      j1=`expr ${i1} % ${nx}`
      if [ $j1 -eq 0 ] ; then
         echo "   </TR><TR>" >> ${file}
      fi

  done   # loop over all files


  echo "    </TR>" >> ${file}
  echo "  </TABLE>" >> ${file}
  DATE=`date`
  echo "  <br><br>Created ${DATE} by (c) ${USER} using <a href=\"createHTMLpage.sh\">createHTMLpage.sh</a>"  >> ${file}
  echo " </BODY>" >> ${file}
  echo "</HTML>" >> ${file}

  cd ..
