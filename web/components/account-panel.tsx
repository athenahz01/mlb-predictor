"use client";

import { FormEvent, useEffect, useState } from "react";
import { createClient } from "@/lib/supabase/client";

export function AccountPanel() {
  const supabase = createClient();
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [signedInEmail, setSignedInEmail] = useState<string | null>(null);
  useEffect(() => {
    supabase?.auth.getUser().then(({ data }) => setSignedInEmail(data.user?.email ?? null));
  }, [supabase]);
  async function signIn(event: FormEvent) {
    event.preventDefault();
    if (!supabase) return;
    const { error } = await supabase.auth.signInWithOtp({
      email,
      options: { emailRedirectTo: `${window.location.origin}/auth/callback` },
    });
    setMessage(error ? error.message : "Check your email for a secure sign-in link.");
  }
  if (!supabase) {
    return (
      <div className="setup-state">
        <strong>Authentication is ready for configuration.</strong>
        <p>Add the Supabase URL and publishable key to enable email magic-link sign-in. No credentials are stored in the repository.</p>
      </div>
    );
  }
  if (signedInEmail) {
    return (
      <div className="profile-card">
        <p className="eyebrow">Signed in</p>
        <h2>{signedInEmail}</h2>
        <p>Your followed teams, players, and display preferences will stay with your account.</p>
        <button className="quiet-button" onClick={() => supabase.auth.signOut()}>Sign out</button>
      </div>
    );
  }
  return (
    <form className="profile-card" onSubmit={signIn}>
      <p className="eyebrow">Email magic link</p>
      <h2>Keep your slate personal</h2>
      <p>Sign in without a password to follow teams and players.</p>
      <label className="field">
        <span>Email address</span>
        <input type="email" required value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" />
      </label>
      <button className="primary-button">Send sign-in link</button>
      {message && <p className="form-message">{message}</p>}
    </form>
  );
}
