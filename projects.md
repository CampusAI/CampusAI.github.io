---
layout: page
title: Projects
permalink: /projects/
---
<style type="text/css">
  .card {
    /* Add shadows to create the "card" effect */
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    transition: 0.3s;
    padding:20px;
    width:100%;
  }
  /* On mouse-over, add a deeper shadow */
  .card:hover {
    box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
  }

  .card_excerpt{
    float:left;
    width: 50%;
  }
  
  .card_video{
  }
  

</style>


<ul>
  {% for post in site.categories.projects %}
      <div class="card">
           <h3><b> {{ post.title }} </b></h3>
           <div class="card_excerpt"> {{post.excerpt}} </div>
           <div class="card_video"> <iframe src="{{post.video}}"/> </div>
      </div>
  {% endfor %}
</ul>
