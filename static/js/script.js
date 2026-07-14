document.addEventListener('DOMContentLoaded', () => {

  // Genre dropdown on the home page navigates straight to /genre/<name>
  const genreSelect = document.getElementById('genreSelect');
  if (genreSelect) {
    genreSelect.addEventListener('change', (e) => {
      const genre = e.target.value;
      if (genre) {
        window.location.href = `/genre/${encodeURIComponent(genre)}`;
      }
    });
  }

  // Show a loading state on the Recommend button while the form submits
  const searchForm = document.getElementById('searchForm');
  const recommendBtn = document.getElementById('recommendBtn');
  if (searchForm && recommendBtn) {
    searchForm.addEventListener('submit', () => {
      const btnText = recommendBtn.querySelector('.btn-text');
      const spinner = recommendBtn.querySelector('.btn-spinner');
      if (btnText) btnText.textContent = 'Searching...';
      if (spinner) spinner.classList.remove('d-none');
      recommendBtn.disabled = true;
    });
  }

  // Slight navbar background shift on scroll for a bit of depth
  const navbar = document.querySelector('.navbar-glass');
  if (navbar) {
    window.addEventListener('scroll', () => {
      if (window.scrollY > 40) {
        navbar.style.boxShadow = '0 8px 24px rgba(0,0,0,0.4)';
      } else {
        navbar.style.boxShadow = 'none';
      }
    });
  }
});
