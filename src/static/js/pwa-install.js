/**
 * PWA Installation Handler
 * Manages app installation prompt and lifecycle
 */

let deferredPrompt;
let isInstalled = false;

// Check if app is already installed
window.addEventListener('DOMContentLoaded', () => {
  checkInstallStatus();
  registerServiceWorker();
  createInstallButton();
  handleAppInstalled();
});

// Register Service Worker
async function registerServiceWorker() {
  if ('serviceWorker' in navigator) {
    try {
      const registration = await navigator.serviceWorker.register('/static/service-worker.js', {
        scope: '/'
      });
      
      console.log('✅ Service Worker registered:', registration.scope);
      
      // Check for updates
      registration.addEventListener('updatefound', () => {
        const newWorker = registration.installing;
        newWorker.addEventListener('statechange', () => {
          if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
            showUpdateNotification();
          }
        });
      });
    } catch (error) {
      console.error('❌ Service Worker registration failed:', error);
    }
  }
}

// Check if app is installed
function checkInstallStatus() {
  // Check if running as installed PWA
  if (window.matchMedia('(display-mode: standalone)').matches || 
      window.navigator.standalone === true) {
    isInstalled = true;
    console.log('✅ Running as installed PWA');
    document.body.classList.add('pwa-installed');
  }
}

// Create install button
function createInstallButton() {
  if (isInstalled) return;
  
  const installContainer = document.createElement('div');
  installContainer.id = 'pwa-install-banner';
  installContainer.className = 'pwa-install-banner hidden';
  installContainer.innerHTML = `
    <div class="pwa-install-content">
      <div class="pwa-install-icon">📱</div>
      <div class="pwa-install-text">
        <h4>Install Mobile App</h4>
        <p>Add to your home screen for quick access</p>
      </div>
      <button id="pwa-install-btn" class="pwa-install-button">
        Install
      </button>
      <button id="pwa-dismiss-btn" class="pwa-dismiss-button">
        ×
      </button>
    </div>
  `;
  
  document.body.appendChild(installContainer);
  
  // Add click handlers
  document.getElementById('pwa-install-btn')?.addEventListener('click', installApp);
  document.getElementById('pwa-dismiss-btn')?.addEventListener('click', dismissInstallBanner);
}

// Capture install prompt event
window.addEventListener('beforeinstallprompt', (e) => {
  console.log('📲 Install prompt available');
  e.preventDefault();
  deferredPrompt = e;
  
  // Show install banner after 3 seconds
  setTimeout(() => {
    showInstallBanner();
  }, 3000);
});

// Show install banner
function showInstallBanner() {
  const banner = document.getElementById('pwa-install-banner');
  if (banner && !isInstalled) {
    banner.classList.remove('hidden');
    banner.classList.add('visible');
  }
}

// Hide install banner
function hideInstallBanner() {
  const banner = document.getElementById('pwa-install-banner');
  if (banner) {
    banner.classList.remove('visible');
    banner.classList.add('hidden');
  }
}

// Dismiss install banner
function dismissInstallBanner() {
  hideInstallBanner();
  // Remember dismissal for 7 days
  localStorage.setItem('pwa-install-dismissed', Date.now().toString());
}

// Install app
async function installApp() {
  if (!deferredPrompt) {
    console.log('❌ No install prompt available');
    return;
  }
  
  hideInstallBanner();
  
  // Show install prompt
  deferredPrompt.prompt();
  
  // Wait for user response
  const { outcome } = await deferredPrompt.userChoice;
  console.log(`User response: ${outcome}`);
  
  if (outcome === 'accepted') {
    console.log('✅ User accepted installation');
  } else {
    console.log('❌ User dismissed installation');
  }
  
  deferredPrompt = null;
}

// Handle app installed
function handleAppInstalled() {
  window.addEventListener('appinstalled', () => {
    console.log('✅ PWA installed successfully!');
    isInstalled = true;
    hideInstallBanner();
    showSuccessNotification('App installed successfully!');
    
    // Track installation (analytics)
    if (typeof trackEvent === 'function') {
      trackEvent('pwa_installed', { method: 'prompt' });
    }
  });
}

// Show update notification
function showUpdateNotification() {
  const notification = document.createElement('div');
  notification.className = 'pwa-update-notification';
  notification.innerHTML = `
    <div class="pwa-update-content">
      <span>🔄 New version available!</span>
      <button onclick="window.location.reload()">Update</button>
    </div>
  `;
  document.body.appendChild(notification);
  
  setTimeout(() => {
    notification.classList.add('visible');
  }, 100);
}

// Show success notification
function showSuccessNotification(message) {
  const notification = document.createElement('div');
  notification.className = 'pwa-success-notification';
  notification.innerHTML = `
    <div class="pwa-success-content">
      <span>✅ ${message}</span>
    </div>
  `;
  document.body.appendChild(notification);
  
  setTimeout(() => {
    notification.classList.add('visible');
  }, 100);
  
  setTimeout(() => {
    notification.classList.remove('visible');
    setTimeout(() => notification.remove(), 300);
  }, 3000);
}

// iOS install instructions
function showIOSInstructions() {
  const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
  if (!isIOS || isInstalled) return;
  
  const modal = document.createElement('div');
  modal.className = 'pwa-ios-modal';
  modal.innerHTML = `
    <div class="pwa-ios-content">
      <h3>Install App on iOS</h3>
      <ol>
        <li>Tap the Share button <span class="ios-share-icon">⎋</span></li>
        <li>Scroll down and tap "Add to Home Screen"</li>
        <li>Tap "Add" to confirm</li>
      </ol>
      <button onclick="this.parentElement.parentElement.remove()">Got it!</button>
    </div>
  `;
  
  document.body.appendChild(modal);
}

// Check if iOS and show instructions after delay
setTimeout(() => {
  if (/iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream && !isInstalled) {
    const dismissed = localStorage.getItem('pwa-install-dismissed');
    if (!dismissed || Date.now() - parseInt(dismissed) > 7 * 24 * 60 * 60 * 1000) {
      showIOSInstructions();
    }
  }
}, 5000);

// Export functions for use in other scripts
window.PWA = {
  install: installApp,
  isInstalled: () => isInstalled,
  showInstallBanner: showInstallBanner,
  hideInstallBanner: hideInstallBanner
};

console.log('📱 PWA Install Handler loaded');

