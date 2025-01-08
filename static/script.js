function fetchPlots() {
    const formData = new FormData(document.getElementById('plot-form'));

    fetch('/get_plot', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.text();
    })
    .then(html => {
        document.getElementById('plot-container').innerHTML = html;
    })
    .catch(error => {
        console.error('There has been a problem with your fetch operation:', error);
    });
}