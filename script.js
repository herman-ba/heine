document.addEventListener('DOMContentLoaded', function() {
  const links = document.querySelectorAll('nav a');
  for (const link of links) {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.scrollIntoView({ behavior: 'smooth' });
      }
    });
  }
});
