const CACHE_NAME = 'github-chatbot-v2';

const ASSETS = [
    './',
    './index.html',
    './style.css',
    './script.js',
    './github_kb.json',
    './manifest.json',
    'https://cdn.jsdelivr.net/npm/marked/marked.min.js'
];

self.addEventListener('install', e => {
    e.waitUntil(
        caches.open(CACHE_NAME).then(cache => {
            console.log("Caching local assets for offline usage...");
            // We use try-catch so that if the CDN is blocked by CORS, it doesn't fail the whole installation!
            return cache.addAll(ASSETS).catch(err => {
                console.warn("Skipping external files due to CORS, caching only local files.", err);
                return cache.addAll([
                    './',
                    './index.html',
                    './style.css',
                    './script.js',
                    './github_kb.json'
                ]);
            });
        })
    );
});

self.addEventListener('activate', e => {
    e.waitUntil(
        caches.keys().then(keys => {
            return Promise.all(keys.map(key => {
                if (key !== CACHE_NAME) {
                    return caches.delete(key);
                }
            }));
        })
    );
});

// NETWORK-FIRST STRATEGY: 
// This fetches from the internet normally (so you always see updates as a developer), 
// but immediately falls back to the saved cache if you lose Wi-Fi!
self.addEventListener('fetch', e => {
    // Ignore internal huggingface requests handling IndexedDB Cache
    if (e.request.url.includes('huggingface.co')) return;

    e.respondWith(
        fetch(e.request).then(response => {
            // Update cache if we had internet
            const responseClone = response.clone();
            caches.open(CACHE_NAME).then(cache => {
                cache.put(e.request, responseClone);
            });
            return response;
        }).catch(async () => {
            // No internet connection! Look into the cache for offline fallback
            const cachedResponse = await caches.match(e.request);
            if (cachedResponse) return cachedResponse;
            if (e.request.url.endsWith('.html') || e.request.url.endsWith('/')) {
                return caches.match('./index.html');
            }
        })
    );
});
