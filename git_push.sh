echo "Pushing to Github..."
git init
git add LICENSE.txt  README.md  butter_bandpass.py  clone.sh  data  data.py  detector.py  detector_utils.py  git_push.sh  images  main.py  modules.sh  requirements.txt  vis_utils.py utils.py
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/Goldenprince8420/ECGAnalyser.git
git push -u origin main
echo "Repository Committed!!"
