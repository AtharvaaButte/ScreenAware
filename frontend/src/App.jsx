import React, { useState, useEffect } from 'react';
import AnimatedCharacters from './components/AnimatedCharacters';
import LoginForm from './components/LoginForm';
import SignupForm from './components/SignupForm';
import LandingPage from './components/LandingPage';
import Home from './components/Home';
import DataForm from './components/DataForm';
import Report from './components/Report';
import TipsPage from './components/TipsPage';
import AnalyticsDashboard from './components/Analytics';
import Navigation from './components/Navigation';
import './App.css';
import { signOut } from 'firebase/auth';
import { auth } from './firebase';

function App() {
  const [isPasswordFocused, setIsPasswordFocused] = useState(false);

  const pathToRoute = (path) => {
    if (!path) return 'landing';
    const p = path.replace(/^\//, '');
    if (p === 'login') return 'login';
    if (p === 'signup') return 'signup';
    if (p === 'dashboard' || p === 'home') return 'dashboard';
    if (p === 'dataform') return 'dataform';
    if (p === 'report') return 'report';
    if (p === 'analytics') return 'analytics';
    if (p === 'tips') return 'tips';
    return 'landing';
  };

  const routeToPath = (r) => {
    if (r === 'login') return '/login';
    if (r === 'signup') return '/signup';
    if (r === 'dashboard') return '/dashboard';
    if (r === 'dataform') return '/dataform';
    if (r === 'report') return '/report';
    if (r === 'analytics') return '/analytics';
    if (r === 'tips') return '/tips';
    return '/';
  };

  const [route, setRoute] = useState(() => pathToRoute(window.location.pathname));

  const navigate = (to, { replace = false } = {}) => {
    const url = routeToPath(to);
    try {
      if (replace) window.history.replaceState({ route: to }, '', url);
      else window.history.pushState({ route: to }, '', url);
    } catch (e) {}
    setRoute(to);
  };

  useEffect(() => {
    const onPop = (event) => {
      const stateRoute = event.state && event.state.route;
      if (stateRoute) setRoute(stateRoute);
      else setRoute(pathToRoute(window.location.pathname));
    };

    window.addEventListener('popstate', onPop);
    window.history.replaceState({ route }, '', routeToPath(route));
    return () => window.removeEventListener('popstate', onPop);
  }, [route]);

  return (
    <div>
      {route !== 'landing' && route !== 'login' && route !== 'signup' && (
        <Navigation 
          onNavigate={navigate} 
          currentRoute={route} 
          onLogout={() => {
            signOut(auth).then(() => {
              navigate('landing', { replace: true });
            }).catch((error) => {
              console.error("Error signing out:", error);
            });
          }}
        />
      )}

      {route === 'landing' ? (
        <LandingPage onNavigate={navigate} />
      ) : route === 'dashboard' ? (
        <Home onNavigate={navigate} />
      ) : route === 'dataform' ? (
        <DataForm onNavigate={navigate} />
      ) : route === 'report' ? (
        <Report onNavigate={navigate} />
      ) : route === 'tips' ? (
        <TipsPage onNavigate={navigate} />
      ) : route === 'analytics' ? (
        <AnalyticsDashboard onNavigate={navigate} />
      ) : (
        <div className="login-container">
          <div className="animation-container">
            <AnimatedCharacters isPasswordFocused={isPasswordFocused} />
          </div>
          <div className="form-container">
            {route === 'login' ? (
              <LoginForm
                onPasswordFocus={() => setIsPasswordFocused(true)}
                onPasswordBlur={() => setIsPasswordFocused(false)}
                onNavigate={navigate}
              />
            ) : (
              <SignupForm onNavigate={navigate} />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
