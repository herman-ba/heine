document.addEventListener('DOMContentLoaded', () => {
  const links = document.querySelectorAll('nav a');
  links.forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      const target = document.querySelector(link.getAttribute('href'));
      if (target) target.scrollIntoView({ behavior: 'smooth' });
    });
  });

  // starfield background
  const canvas = document.getElementById('bg');
  const ctx = canvas.getContext('2d');
  let width, height;
  function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  }
  window.addEventListener('resize', resize);
  resize();

  const stars = Array.from({ length: 200 }, () => ({
    x: Math.random() * width - width / 2,
    y: Math.random() * height - height / 2,
    z: Math.random() * width
  }));

  function updateStars() {
    ctx.fillStyle = 'rgba(15,32,39,0.4)';
    ctx.fillRect(0, 0, width, height);
    for (const star of stars) {
      star.z -= 2;
      if (star.z <= 0) {
        star.z = width;
        star.x = Math.random() * width - width / 2;
        star.y = Math.random() * height - height / 2;
      }
      const k = 128 / star.z;
      const x = star.x * k + width / 2;
      const y = star.y * k + height / 2;
      const size = (1 - star.z / width) * 3;
      if (x >= 0 && x < width && y >= 0 && y < height) {
        ctx.fillStyle = 'rgba(255,255,255,' + (1 - star.z / width) + ')';
        ctx.fillRect(x, y, size, size);
      }
    }
    requestAnimationFrame(updateStars);
  }
  updateStars();

  // 3D card hover
  const cards = document.querySelectorAll('.project');
  cards.forEach(card => {
    card.addEventListener('mousemove', e => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const rotateX = ((y / rect.height) - 0.5) * -20;
      const rotateY = ((x / rect.width) - 0.5) * 20;
      card.style.transform = `rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
    });
    card.addEventListener('mouseleave', () => {
      card.style.transform = '';
    });
  });
});
