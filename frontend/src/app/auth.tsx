import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  type PropsWithChildren,
} from "react";
import { useQueryClient } from "@tanstack/react-query";

import { getMe } from "../api/client";
import type { Identity, Permission } from "../api/types";

const TOKEN_KEY = "rag-kb.demo-token";

type AuthState = {
  token: string | null;
  identity: Identity | null;
  checking: boolean;
  error: string | null;
  session: number;
  connect: (token: string) => Promise<Identity>;
  disconnect: () => void;
  hasPermission: (permission: Permission) => boolean;
};

const AuthContext = createContext<AuthState | null>(null);

function storedToken(): string | null {
  if (typeof window === "undefined") return null;
  return sessionStorage.getItem(TOKEN_KEY);
}

export function AuthProvider({ children }: PropsWithChildren) {
  const queryClient = useQueryClient();
  const [token, setToken] = useState<string | null>(storedToken);
  const [identity, setIdentity] = useState<Identity | null>(null);
  const [checking, setChecking] = useState(Boolean(token));
  const [error, setError] = useState<string | null>(null);
  const [session, setSession] = useState(0);
  const generation = useRef(0);

  const reset = () => {
    generation.current += 1;
    void queryClient.cancelQueries();
    queryClient.clear();
    sessionStorage.removeItem(TOKEN_KEY);
    setToken(null);
    setIdentity(null);
    setChecking(false);
    setSession((value) => value + 1);
  };

  const connect = async (rawToken: string): Promise<Identity> => {
    const nextToken = rawToken.trim();
    if (!nextToken) throw new Error("请输入演示令牌");
    reset();
    const current = generation.current;
    setChecking(true);
    setError(null);
    try {
      const nextIdentity = await getMe(nextToken);
      if (current !== generation.current) throw new Error("身份已切换");
      sessionStorage.setItem(TOKEN_KEY, nextToken);
      setToken(nextToken);
      setIdentity(nextIdentity);
      setSession((value) => value + 1);
      return nextIdentity;
    } catch (cause) {
      if (current === generation.current) {
        setError(cause instanceof Error ? cause.message : "身份验证失败");
      }
      throw cause;
    } finally {
      if (current === generation.current) setChecking(false);
    }
  };

  useEffect(() => {
    const initial = storedToken();
    if (!initial) return;
    const current = generation.current;
    getMe(initial)
      .then((nextIdentity) => {
        if (current === generation.current) setIdentity(nextIdentity);
      })
      .catch(() => {
        if (current === generation.current) reset();
      })
      .finally(() => {
        if (current === generation.current) setChecking(false);
      });
  }, []);

  return (
    <AuthContext.Provider
      value={{
        token,
        identity,
        checking,
        error,
        session,
        connect,
        disconnect: reset,
        hasPermission: (permission) => Boolean(identity?.permissions.includes(permission)),
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthState {
  const value = useContext(AuthContext);
  if (!value) throw new Error("AuthProvider is required");
  return value;
}
