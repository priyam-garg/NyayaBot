import { createContext, useContext, useEffect, useState } from "react";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem("nyayabot_token");
    const cachedUser = localStorage.getItem("nyayabot_user");
    if (token && cachedUser) {
      try {
        setUser(JSON.parse(cachedUser));
      } catch {
        localStorage.removeItem("nyayabot_user");
      }
    }
    setReady(true);
  }, []);

  const login = (token, userObj) => {
    localStorage.setItem("nyayabot_token", token);
    localStorage.setItem("nyayabot_user", JSON.stringify(userObj));
    setUser(userObj);
  };

  const logout = () => {
    localStorage.removeItem("nyayabot_token");
    localStorage.removeItem("nyayabot_user");
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, ready, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
