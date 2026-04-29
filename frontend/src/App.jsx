import { Navigate, Route, Routes } from "react-router-dom";
import Login from "./pages/Login.jsx";
import Signup from "./pages/Signup.jsx";
import Chat from "./pages/Chat.jsx";
import Compare from "./pages/Compare.jsx";
import { useAuth } from "./context/AuthContext.jsx";

function Protected({ children }) {
  const { user, ready } = useAuth();
  if (!ready) return null;
  if (!user) return <Navigate to="/login" replace />;
  return children;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />
      <Route
        path="/chat"
        element={
          <Protected>
            <Chat />
          </Protected>
        }
      />
      <Route
        path="/compare"
        element={
          <Protected>
            <Compare />
          </Protected>
        }
      />
      <Route path="*" element={<Navigate to="/chat" replace />} />
    </Routes>
  );
}
