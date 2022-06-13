echo "Pushing to Github..."
git init
git add LICENSE.txt  README.md  clone.sh  git_push.sh  main.py
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/Goldenprince8420/ECGAnalyser.git
git push -u origin main
echo "Repository Committed!!"
