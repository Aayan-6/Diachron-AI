import { Suspense, lazy } from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import AppShell from './components/layout/AppShell';
import { Spinner } from './components/ui/Feedback';
import { ErrorBoundary } from './components/ui/ErrorBoundary';
import { AppStateProvider } from './context/AppState';
import Workspace from './pages/Workspace';

/*
 * The workspace is the landing surface for real use, so it ships in the entry
 * chunk. Everything else is split out — most importantly the marketing page,
 * which drags in three.js and @react-three for its 3D hero. Loading a WebGL
 * scene graph just to open the chat was most of the bundle.
 */
const LandingPage = lazy(() => import('./pages/LandingPage'));
const Login = lazy(() => import('./pages/Login'));
const MapPage = lazy(() => import('./pages/MapPage'));
const ProfilePage = lazy(() => import('./pages/ProfilePage'));
const SettingsPage = lazy(() => import('./pages/SettingsPage'));

const RouteFallback = () => (
  <div className="flex h-full min-h-[60vh] w-full items-center justify-center bg-ground">
    <Spinner size={20} className="text-accent" />
  </div>
);

const App = () => (
  <AppStateProvider>
    <BrowserRouter>
      <ErrorBoundary fallbackTitle="Diachron AI UI Encountered an Issue">
        <Suspense fallback={<RouteFallback />}>
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/login" element={<Login mode="login" />} />
            <Route path="/signup" element={<Login mode="signup" />} />

            {/* Everything inside the product shell. */}
            <Route element={<AppShell />}>
              <Route path="/console" element={<Workspace />} />
              <Route path="/map" element={<MapPage />} />
              <Route path="/profile" element={<ProfilePage />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Route>

            {/* History is a panel in the shell now, not a page. */}
            <Route path="/history" element={<Navigate to="/console" replace />} />

            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </ErrorBoundary>
    </BrowserRouter>
  </AppStateProvider>
);

export default App;
