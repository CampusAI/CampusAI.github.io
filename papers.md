---
layout: page
title: Papers
permalink: /papers/
---

# RL Algorithms

# Hierarchical RL
{% for item in site.papers %}
{% if item.category != "hierarchical" %}
    {% continue %}
{% endif %}
[{{item.title}}]({{item.permalink}})
{% endfor %}

# Other
{% for item in site.papers %}
{% if item.category != "other" %}
    {% continue %}
{% endif %}
[{{item.title}}]({{item.permalink}})
{% endfor %}
