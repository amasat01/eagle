{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}

{% for item in names %}
   {{ item }}
{%- endfor %}
