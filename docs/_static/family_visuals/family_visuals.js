/* Closes the data tables that are open in the page source so that a reader without script sees them. */
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("details[data-nojs]").forEach(function (d) { d.removeAttribute("open"); });
});
