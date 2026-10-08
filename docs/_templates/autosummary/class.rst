{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods and methods|length > 1 %}
   .. rubric:: {{ _('Methods') }}

   {% set allowed_specials = ['__init__', '__enter__', '__exit__', '__iter__', '__next__', '__getitem__', '__setitem__', '__len__', '__array__'] %}

   .. autosummary::
      :signatures: short
   {% for item in methods|sort %}
   {% if item not in inherited_members and (item in allowed_specials or not item.startswith('_')) %}
      ~{{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}

   {% set visible_inherited = [] %}
   {% for item in methods %}
      {% if item in inherited_members and (item in allowed_specials or not item.startswith('_')) %}
         {% set _ = visible_inherited.append(item) %}
      {% endif %}
   {% endfor %}


   {% if visible_inherited %}
   .. rubric:: {{ _('Inherited methods and attributes') }}

   .. autosummary::
      :signatures: short
   {% for item in visible_inherited %}
      ~{{ name }}.{{ item }}
   {%- endfor %}

   {% endif %}
   {% endif %}
   {% endblock %}
