<h1>Semantic Search DB</h1>
<img src="https://media.dev.to/cdn-cgi/image/width=1000,height=420,fit=cover,gravity=auto,format=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fikqfpsu3jd60em4s0ztn.png">


## <img align= center width=50px height=50px src="https://thumbs.gfycat.com/HeftyDescriptiveChimneyswift-size_restricted.gif"> Table of Contents

- <a href ="#Overview"> Overview</a>
- <a href ="#started"> Get Started</a>
- <a href ="#modules">  Methods</a>
  - IVF
  - LSH
  - PQ
- <a href ="#contributors"> 🧑 Contributors</a>
- <a href ="#license"> 🔒 License</a>

## <img align="center"  width =50px  height =50px src="https://em-content.zobj.net/source/animated-noto-color-emoji/356/waving-hand_1f44b.gif"> Overview <a id = "Overview"></a>
Given the embedding of the search query we can efficent get the top matching k results form DB with 20M document.The objective of this project is to design and implement an indexing system for a
semantic search database.


## <img  align= center width=50px height=50px src="https://cdn.pixabay.com/animation/2022/07/31/06/27/06-27-17-124_512.gif">Get Started <a id = "started"></a>
### Infernce Mode
```
# Check Final Notebook
https://github.com/ZiadSheriif/sematic_search_DB/blob/main/Evaluate_ADB_Project.ipynb
```
### Run Locally
```
# Clone Repo
git clone https://github.com/ZiadSheriif/sematic_search_DB.git

# Install dependencies
pip install -r requirements.txt

# Run Indexer
$ python ./src/evaluation.py
```


## <img  align= center width=50px height=50px src="https://media3.giphy.com/media/l0G372BYKnKuBkKxjo/giphy.gif?cid=6c09b952k9s08y3588aqm3f31dpyz9u0qnfe0gh5s8tyj0l4&ep=v1_stickers_related&rid=giphy.gif&ct=s">Methods<a id = "started"></a>
### IVF 
This is out final Approach with Some Enhancements 
1. Changed MiniBatchKMeans to regular KMeans
2. We calculate initial centroids with just the first chunk of data
3. Introduced parallel processing for different regions
<img src="https://miro.medium.com/v2/resize:fit:786/format:webp/1*CSwHz4IlVnqufq1QdmMtVg.png">

### LSH
<img src="https://cdn.sanity.io/images/vr8gru94/production/862f88182a796eb16942c47d93ee03ba4cdaee4d-1920x1080.png">

### PQ
<img src="https://miro.medium.com/v2/resize:fit:786/format:webp/1*98eO9hCC3Wzp8AURuZT-NA.png">
### PQ_LSH

<!-- Contributors -->
## <img  align= center width=50px height=50px src="https://media1.giphy.com/media/WFZvB7VIXBgiz3oDXE/giphy.gif?cid=6c09b952tmewuarqtlyfot8t8i0kh6ov6vrypnwdrihlsshb&rid=giphy.gif&ct=s"> Contributors <a id = "contributors"></a>

<!-- Contributors list -->
<table align="center" >
  <tr>
    <td align="center"><a href="https://github.com/BasmaElhoseny01"><img src="https://avatars.githubusercontent.com/u/72309546?v=4" width="150px;" alt=""/><br /><sub><b>Basma Elhoseny</b></sub></a><br /></td>
    <td align="center"><a href=""><img src="" width="150px;" alt=""/><br /><sub><b>Yasmine Ghanem</b></sub></a><br /></td>
    <td align="center"><a href="" ><img src="https://avatars.githubusercontent.com/u/68201932?v=4" width="150px;" alt=""/><br /><sub><b>Mohab Zaghloul</b></sub></a><br />
    </td>
     <td align="center"><a href="https://github.com/YasminElgendi"><img src="https://avatars.githubusercontent.com/u/54359829?v=4" width="150px;" alt=""/><br /><sub><b>Yasmin Elgendi</b></sub></a><br /></td>
  </tr>
</table>

<br>

## <img align="center"  width =50px  height =50px src="https://images.squarespace-cdn.com/content/v1/5c88c50af4e5316a44e9f34e/1639666090540-WIW96612QF3IQPGQXPD3/giphy+%284%29.gif"> FeedBack <a id = "feedback"></a>
> If you have any feedback, please reach out to us at mohabmohamedmohamedzaghloul@gmail.com.

<br>


## <img  align= center width=50px height=50px src="https://media1.giphy.com/media/ggoKD4cFbqd4nyugH2/giphy.gif?cid=6c09b9527jpi8kfxsj6eswuvb7ay2p0rgv57b7wg0jkihhhv&rid=giphy.gif&ct=s"> License <a id = "license"></a>
This software is licensed under MIT License, See [License](https://github.com/BasmaElhoseny01/Hand-Gesture-Recognition/blob/main/LICENSE) for more information ©Basma Elhoseny.
