'use client';

import { useState, useRef, useEffect } from 'react';
import {
    Send, Image as ImageIcon, Plus, Menu, User, Cpu, X, Zap,
    MessageSquare, Trash2, Settings, LogOut, Sparkles, ChevronRight, UserCircle
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import styles from './ChatInterface.module.css';

type Message = {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    image?: string;
};

type ChatSession = {
    id: string;
    title: string;
    messages: Message[];
    createdAt: number;
};

type UserProfile = {
    name: string;
    email: string;
};

// Component to simulate typing effect
const Typewriter = ({ text, onComplete }: { text: string, onComplete?: () => void }) => {
    const [displayedText, setDisplayedText] = useState('');

    useEffect(() => {
        let index = 0;
        const intervalId = setInterval(() => {
            setDisplayedText((prev) => prev + text.charAt(index));
            index++;
            if (index === text.length) {
                clearInterval(intervalId);
                onComplete && onComplete();
            }
        }, 15); // Adjust speed here (lower = faster)
        return () => clearInterval(intervalId);
    }, [text, onComplete]);

    return <span style={{ whiteSpace: 'pre-wrap' }}>{displayedText}</span>;
};


export default function ChatInterface() {
    const [sessions, setSessions] = useState<ChatSession[]>([]);
    const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
    const [input, setInput] = useState('');
    const [selectedImage, setSelectedImage] = useState<File | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);

    // Typing effect state
    const [isTyping, setIsTyping] = useState(false);

    // Settings / Auth
    const [showSettings, setShowSettings] = useState(false);
    const [showPreferences, setShowPreferences] = useState(false);
    const [currentTheme, setCurrentTheme] = useState('system');
    const [showLogin, setShowLogin] = useState(false); // Restored login modal
    const [user, setUser] = useState<UserProfile | null>(null);
    const [tempName, setTempName] = useState('');

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    // Load state
    useEffect(() => {
        const savedSessions = localStorage.getItem('repairgpt_sessions');
        if (savedSessions) {
            try {
                const parsed = JSON.parse(savedSessions);
                setSessions(parsed);
                if (parsed.length > 0) setCurrentSessionId(parsed[0].id);
            } catch (e) {
                console.error("Failed to load sessions", e);
            }
        }

        const savedUser = localStorage.getItem('repairgpt_user');
        if (savedUser) {
            setUser(JSON.parse(savedUser));
        } else {
            // Prompt for login if no user
            setShowLogin(true);
        }

        // Apply theme
        // Apply theme
        const savedTheme = localStorage.getItem('repairgpt_theme');
        if (savedTheme) {
            setCurrentTheme(savedTheme);
            if (savedTheme === 'dark') document.documentElement.classList.add('dark');
            else document.documentElement.classList.remove('dark');
        } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
            document.documentElement.classList.add('dark');
        }
    }, []);

    // Persist sessions
    useEffect(() => {
        localStorage.setItem('repairgpt_sessions', JSON.stringify(sessions));
    }, [sessions]);

    // Persist user
    useEffect(() => {
        if (user) localStorage.setItem('repairgpt_user', JSON.stringify(user));
        else localStorage.removeItem('repairgpt_user');
    }, [user]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    const currentSession = sessions.find(s => s.id === currentSessionId);
    const messages = currentSession ? currentSession.messages : [];

    useEffect(() => {
        scrollToBottom();
    }, [messages, isLoading, isTyping, currentSessionId]);

    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 200) + 'px';
        }
    }, [input]);

    const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setSelectedImage(file);
            const url = URL.createObjectURL(file);
            setImagePreview(url);
        }
    };

    const clearImage = () => {
        setSelectedImage(null);
        setImagePreview(null);
    };

    const createNewChat = () => {
        const newId = Date.now().toString();
        const newSession: ChatSession = {
            id: newId,
            title: 'New Chat',
            messages: [],
            createdAt: Date.now()
        };
        setSessions(prev => [newSession, ...prev]);
        setCurrentSessionId(newId);
        setIsSidebarOpen(false);
        return newId;
    };

    const deleteChat = (e: React.MouseEvent, id: string) => {
        e.stopPropagation();
        const newSessions = sessions.filter(s => s.id !== id);
        setSessions(newSessions);
        if (currentSessionId === id) {
            setCurrentSessionId(newSessions.length > 0 ? newSessions[0].id : null);
        }
    };

    const handleLogin = () => {
        if (!tempName.trim()) return;
        setUser({ name: tempName, email: `${tempName.toLowerCase().replace(/\s/g, '')}@example.com` });
        setShowLogin(false);
    };

    const handleLogout = () => {
        setUser(null);
        setTempName('');
        setShowSettings(false); // Close menu
        setShowLogin(true); // Re-open login prompt
    };

    // Helper to convert file to base64 for persistence
    const fileToBase64 = (file: File): Promise<string> => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result as string);
            reader.onerror = error => reject(error);
        });
    };

    const handleSubmit = async (e?: React.FormEvent) => {
        e?.preventDefault();
        if ((!input.trim() && !selectedImage) || isLoading || isTyping) return;

        let activeId = currentSessionId;
        if (!activeId) activeId = createNewChat();

        const currentInput = input;

        // Convert image to Base64 for persistent storage (Blob URLs die on reload)
        let imageBase64: string | undefined = undefined;
        if (selectedImage) {
            try {
                imageBase64 = await fileToBase64(selectedImage);
            } catch (err) {
                console.error("Failed to convert image", err);
            }
        }

        const newMessage: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: currentInput,
            image: imageBase64 // Use Base64 instead of Blob URL
        };

        setSessions(prev => prev.map(session => {
            if (session.id === activeId) {
                const newTitle = session.messages.length === 0
                    ? (currentInput.length > 30 ? currentInput.substring(0, 30) + '...' : (currentInput || 'Image Upload'))
                    : session.title;
                return {
                    ...session,
                    title: newTitle,
                    messages: [...session.messages, newMessage]
                };
            }
            return session;
        }));

        setInput('');
        if (textareaRef.current) textareaRef.current.style.height = 'auto';
        setIsLoading(true);

        // Keep local image for upload, clear strictly after
        const imageToUpload = selectedImage;
        clearImage(); // Clear preview immediately for better UX

        try {
            const formData = new FormData();
            formData.append('message', currentInput);
            if (imageToUpload) formData.append('image', imageToUpload);

            const res = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();

            // Instead of adding immediately, we start the typing effect
            setIsLoading(false);
            setIsTyping(true);

            // Temporary holding for the full message
            const botMessageContent = data.reply;

            const botMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: botMessageContent
            };

            setSessions(prev => prev.map(s => s.id === activeId ? ({ ...s, messages: [...s.messages, botMessage] }) : s));

        } catch (err) {
            console.error(err);
            const errorMsg: Message = { id: Date.now().toString(), role: 'assistant', content: "An error occurred. Please try again." };
            setSessions(prev => prev.map(s => s.id === activeId ? ({ ...s, messages: [...s.messages, errorMsg] }) : s));
            setIsLoading(false);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    return (
        <div className={styles.layout}>
            {/* Mobile Header */}
            <div className={styles.mobileHeader}>
                <button onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
                    <Menu className="w-5 h-5" />
                </button>
                <span className="font-semibold text-sm">RepairGPT</span>
                <Plus className="w-5 h-5" onClick={createNewChat} />
            </div>

            {/* Sidebar */}
            <div className={`${styles.sidebar} ${isSidebarOpen ? styles.open : ''}`}>
                <div className={styles.sidebarHeader}>
                    <button className={styles.newChatBtn} onClick={createNewChat}>
                        <Plus className="w-4 h-4" />
                        <span>New chat</span>
                    </button>
                </div>

                <div className={styles.historyList}>
                    <div className={styles.historyGroupTitle}>Recent</div>
                    {sessions.length === 0 && (
                        <p className="text-gray-400 text-xs px-3 italic">No chat history</p>
                    )}
                    {sessions.map(session => (
                        <div
                            key={session.id}
                            className={`${styles.historyItem} ${session.id === currentSessionId ? styles.active : ''}`}
                            onClick={() => { setCurrentSessionId(session.id); setIsSidebarOpen(false); }}
                        >
                            <MessageSquare className="w-4 h-4 shrink-0 opacity-70" />
                            <span className="truncate flex-1">{session.title}</span>
                            <button
                                className={styles.deleteBtn}
                                onClick={(e) => deleteChat(e, session.id)}
                            >
                                <Trash2 className="w-3 h-3" />
                            </button>
                        </div>
                    ))}
                </div>

                <div className="mt-auto relative">
                    <div className={styles.userProfile} onClick={() => setShowSettings(!showSettings)}>
                        <div className={styles.userAvatarPlaceholder}>
                            <User className="w-5 h-5" />
                        </div>
                        <div className={styles.userInfo}>
                            <span className={styles.userName}>{user ? user.name : 'Sign In'}</span>
                            <span className={styles.userEmail}>{user ? 'Free Plan' : 'Guest'}</span>
                        </div>
                        <Settings className={`${styles.settingsIcon} w-4 h-4`} />
                    </div>
                </div>
            </div>

            {/* Mobile Overlay */}
            {isSidebarOpen && <div className={styles.modalOverlay} onClick={() => setIsSidebarOpen(false)} style={{ zIndex: 45 }} />}

            {/* Popover Menu (Moved outside Sidebar to avoid clipping/transform issues) */}
            <AnimatePresence>
                {showSettings && (
                    <>
                        {/* Invisible backdrop to close menu on click outside */}
                        <div
                            className="fixed inset-0 z-[90]"
                            onClick={() => setShowSettings(false)}
                        />
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95, y: 10 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.95, y: 10 }}
                            transition={{ duration: 0.1 }}
                            className={styles.popoverMenu}
                            style={{
                                position: 'fixed',
                                bottom: '80px',
                                left: '10px',
                                width: '240px',
                                zIndex: 100,
                                transformOrigin: 'bottom center'
                            }}
                        >
                            {user ? (
                                <>
                                    <div className={styles.menuHeader}>
                                        <div className={styles.menuName}>{user.name}</div>
                                        <div className={styles.menuHandle}>@{user.name.toLowerCase().replace(/\s/g, '')}</div>
                                    </div>

                                    <div className={styles.menuItem}>
                                        <Sparkles className={styles.itemIcon} />
                                        <span>Upgrade plan</span>
                                    </div>
                                    <div className={styles.menuItem}>
                                        <UserCircle className={styles.itemIcon} />
                                        <span>Personalization</span>
                                    </div>
                                    <div
                                        className={styles.menuItem}
                                        onClick={() => {
                                            setShowSettings(false); // Close menu
                                            setShowPreferences(true); // Open settings modal
                                        }}
                                    >
                                        <Settings className={styles.itemIcon} />
                                        <span>Settings</span>
                                    </div>

                                    <div className={styles.menuSeparator} />

                                    <div className={styles.menuItem}>
                                        <div className={styles.itemIcon}><Zap className="w-4 h-4" /></div>
                                        <span>Help</span>
                                    </div>
                                    <div className={`${styles.menuItem} text-red-600 hover:text-red-700 hover:bg-red-50`} onClick={handleLogout}>
                                        <LogOut className={`${styles.itemIcon} text-red-500`} />
                                        <span>Log out</span>
                                    </div>
                                </>
                            ) : (
                                <div className="p-4">
                                    <p className="text-gray-500 mb-4 text-sm">Please sign in to access settings.</p>
                                    <input
                                        autoFocus
                                        type="text"
                                        placeholder="Enter your name"
                                        className="w-full p-2 border rounded mb-2 text-sm"
                                        value={tempName}
                                        onChange={(e) => setTempName(e.target.value)}
                                        onKeyDown={(e) => e.key === 'Enter' && handleLogin()}
                                    />
                                    <button
                                        className="w-full bg-black text-white p-2 rounded text-sm font-medium"
                                        onClick={handleLogin}
                                    >
                                        Sign In
                                    </button>
                                </div>
                            )}
                        </motion.div>
                    </>
                )}
            </AnimatePresence>

            {/* Main Chat */}
            <div className={styles.mainArea}>
                <AnimatePresence mode='wait'>
                    {!currentSession || messages.length === 0 ? (
                        <motion.div
                            key="intro"
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 1.05 }}
                            transition={{ duration: 0.3 }}
                            className={styles.intro}
                        >
                            <Sparkles className={styles.logoIcon} />
                            <h1 className={styles.gradientText}>How can I help you today?</h1>

                            <div className={styles.grid}>
                                <div className={styles.card} onClick={() => setInput("Diagnose my laptop screen issue")}>
                                    <Zap className={styles.cardIcon} />
                                    <div className={styles.cardTitle}>Quick Diagnosis</div>
                                    <div className={styles.cardDesc}>Identify common hardware faults instantly</div>
                                </div>
                                <div className={styles.card} onClick={() => setInput("Identify this comprehensive circuit")}>
                                    <Cpu className={styles.cardIcon} />
                                    <div className={styles.cardTitle}>Component ID</div>
                                    <div className={styles.cardDesc}>Recognize complex electronic parts from photos</div>
                                </div>
                            </div>
                        </motion.div>
                    ) : (
                        <div className={styles.chatContainer}>
                            {messages.map((msg, idx) => {
                                const isLastMessage = idx === messages.length - 1;
                                const shouldType = isLastMessage && msg.role === 'assistant' && isTyping;

                                return (
                                    <motion.div
                                        key={msg.id}
                                        initial={{ opacity: 0, y: 10 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ duration: 0.3 }}
                                        className={`${styles.messageWrapper} ${msg.role === 'assistant' ? styles.assistant : ''}`}
                                    >
                                        <div className={styles.messageContent}>
                                            <div className={`${styles.avatar} ${msg.role === 'assistant' ? styles.botAvatar : styles.userAvatar}`}>
                                                {msg.role === 'assistant' ? <Sparkles className="w-4 h-4" /> : <UserCircle className="w-5 h-5" />}
                                            </div>
                                            <div className={styles.text}>
                                                <div className={styles.msgTitle}>{msg.role === 'assistant' ? 'RepairGPT' : (user?.name || 'You')}</div>
                                                {msg.image && <img src={msg.image} alt="User Upload" className={styles.uploadedImage} />}

                                                {shouldType ? (
                                                    <Typewriter
                                                        text={msg.content}
                                                        onComplete={() => setIsTyping(false)}
                                                    />
                                                ) : (
                                                    <span style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</span>
                                                )}
                                            </div>
                                        </div>
                                    </motion.div>
                                );
                            })}

                            {isLoading && (
                                <div className={`${styles.messageWrapper} ${styles.assistant}`}>
                                    <div className={styles.messageContent}>
                                        <div className={`${styles.avatar} ${styles.botAvatar}`}>
                                            <Sparkles className="w-4 h-4" />
                                        </div>
                                        <div className={styles.text}>
                                            <div className={styles.msgTitle}>RepairGPT</div>
                                            <div className={styles.typingIndicator}>
                                                <div className={styles.typingDot}></div>
                                                <div className={styles.typingDot}></div>
                                                <div className={styles.typingDot}></div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}
                            <div ref={messagesEndRef} />
                        </div>
                    )}
                </AnimatePresence>

                {/* Input Area */}
                <div className={styles.inputContainer}>
                    <div className={styles.inputBox}>
                        {imagePreview && (
                            <div className={styles.previewArea}>
                                <div className="relative inline-block overflow-hidden rounded-lg">
                                    <img src={imagePreview} className={styles.previewImage} alt="Preview" />
                                    <button onClick={clearImage} className="absolute -top-2 -right-2 bg-gray-900 text-white rounded-full p-0.5 border border-white z-20">
                                        <X className="w-3 h-3" />
                                    </button>
                                </div>
                            </div>
                        )}
                        <div className={styles.inputWrapper}>
                            <label className={styles.actionBtn} title="Upload Image">
                                <input type="file" accept="image/*" onChange={handleImageSelect} className="hidden" style={{ display: 'none' }} />
                                <ImageIcon className="w-5 h-5" />
                            </label>
                            <textarea
                                ref={textareaRef}
                                rows={1}
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder="Message RepairGPT..."
                                className={styles.textArea}
                            />
                            <button
                                onClick={() => handleSubmit()}
                                disabled={(!input.trim() && !selectedImage) || isLoading || isTyping}
                                className={styles.sendBtn}
                            >
                                <Send className="w-4 h-4" />
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Mobile Overlay */}
            {isSidebarOpen && <div className={styles.modalOverlay} onClick={() => setIsSidebarOpen(false)} style={{ zIndex: 45 }} />}

            {/* Login Modal (Restored Center Modal) */}
            <AnimatePresence>
                {showLogin && (
                    <div className={styles.modalOverlay}>
                        <motion.div
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.9, opacity: 0 }}
                            className={styles.modal}
                        >
                            <div className={styles.modalHeader}>
                                <h2 className={styles.modalTitle}>Welcome back</h2>
                                <button className={styles.closeBtn} onClick={() => setShowLogin(false)}>
                                    {/* Allow closing if they really want to be guest, or make it mandatory? Let's allow closing */}
                                    <X className="w-5 h-5" />
                                </button>
                            </div>

                            <div>
                                <p className="text-gray-500 mb-6">Please enter your details to continue.</p>
                                <input
                                    autoFocus
                                    type="text"
                                    placeholder="Your Name"
                                    className={styles.modalInput}
                                    value={tempName}
                                    onChange={(e) => setTempName(e.target.value)}
                                    onKeyDown={(e) => e.key === 'Enter' && handleLogin()}
                                />
                                <button className={styles.primaryBtn} onClick={handleLogin}>
                                    Continue
                                </button>
                            </div>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>

            {/* Actual Settings (Preferences) Modal */}
            <AnimatePresence>
                {showPreferences && (
                    <div className={styles.modalOverlay}>
                        <motion.div
                            initial={{ scale: 0.95, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.95, opacity: 0 }}
                            className={styles.settingsModal}
                        >
                            {/* Close Button at top right of modal */}
                            <button className={styles.settingsCloseBtn} onClick={() => setShowPreferences(false)}>
                                <X className="w-5 h-5" />
                            </button>

                            {/* Sidebar */}
                            <div className={styles.settingsSidebar}>
                                <div className={`${styles.settingsSidebarItem} ${styles.active}`}>
                                    <Settings className="w-4 h-4" />
                                    <span>General</span>
                                </div>
                                <div className={styles.settingsSidebarItem}>
                                    <UserCircle className="w-4 h-4" />
                                    <span>Personalization</span>
                                </div>
                                <div className={styles.settingsSidebarItem}>
                                    <Zap className="w-4 h-4" />
                                    <span>Data controls</span>
                                </div>
                            </div>

                            <div className={styles.settingsContent}>
                                <h2 className={styles.settingsSectionTitle}>General</h2>

                                {/* Appearance */}
                                <div className={styles.settingsRow}>
                                    <span className={styles.settingsLabel}>Appearance</span>
                                    <div className={styles.selectWrapper}>
                                        <select
                                            className={styles.selectInput}
                                            value={currentTheme}
                                            onChange={(e) => {
                                                const val = e.target.value;
                                                setCurrentTheme(val);
                                                if (val === 'system') {
                                                    localStorage.removeItem('repairgpt_theme');
                                                    if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
                                                        document.documentElement.classList.add('dark');
                                                    } else {
                                                        document.documentElement.classList.remove('dark');
                                                    }
                                                } else {
                                                    localStorage.setItem('repairgpt_theme', val);
                                                    if (val === 'dark') document.documentElement.classList.add('dark');
                                                    else document.documentElement.classList.remove('dark');
                                                }
                                            }}
                                        >
                                            <option value="system">System</option>
                                            <option value="dark">Dark</option>
                                            <option value="light">Light</option>
                                        </select>
                                        <ChevronRight className={`${styles.selectChevron} rotate-90`} />
                                    </div>
                                </div>

                                {/* Accent Color */}
                                <div className={styles.settingsRow}>
                                    <span className={styles.settingsLabel}>Accent color</span>
                                    <div className={styles.settingValue}>
                                        <div className="w-3 h-3 rounded-full bg-gray-400"></div>
                                        <span>Default</span>
                                        <ChevronRight className="w-4 h-4 rotate-90 opacity-50" />
                                    </div>
                                </div>

                                {/* Language */}
                                <div className={styles.settingsRow}>
                                    <span className={styles.settingsLabel}>Language</span>
                                    <div className={styles.settingValue}>
                                        <span>Auto-detect</span>
                                        <ChevronRight className="w-4 h-4 rotate-90 opacity-50" />
                                    </div>
                                </div>

                                {/* Spoken Language */}
                                <div className={styles.settingsRow}>
                                    <span className={styles.settingsLabel}>Spoken language</span>
                                    <div className={styles.settingValue}>
                                        <span>Auto-detect</span>
                                        <ChevronRight className="w-4 h-4 rotate-90 opacity-50" />
                                    </div>
                                </div>

                                <div className="text-xs text-gray-500 mb-4 -mt-2.5 ml-0 leading-relaxed">
                                    For best results, select the language you mainly speak. If it's not listed, it may still be supported via auto-detection.
                                </div>

                                {/* Voice */}
                                <div className={styles.settingsRow}>
                                    <span className={styles.settingsLabel}>Voice</span>
                                    <div className="flex items-center gap-3">
                                        <button className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
                                            <div className="w-0 h-0 border-t-[5px] border-t-transparent border-l-[8px] border-l-black dark:border-l-white border-b-[5px] border-b-transparent ml-0.5"></div>
                                        </button>
                                        <div className={styles.settingValue}>
                                            <span>Vale</span>
                                            <ChevronRight className="w-4 h-4 rotate-90 opacity-50" />
                                        </div>
                                    </div>
                                </div>

                                {/* Separate Voice Toggle */}
                                <div className={styles.settingsRow}>
                                    <div className="flex flex-col gap-1 max-w-[80%]">
                                        <span className={styles.settingsLabel}>Separate Voice</span>
                                        <span className="text-xs text-gray-500 leading-snug">Keep Chat Voice in a separate full screen.</span>
                                    </div>
                                    <label className={styles.toggleWrapper}>
                                        <input type="checkbox" className={styles.toggleSwitchInput} />
                                        <div className={styles.toggleSwitch}>
                                            <div className={styles.toggleSwitchThumb}></div>
                                        </div>
                                    </label>
                                </div>

                                {/* Delete moved to Data Controls or just kept hidden for now to match screenshot accurately */}
                            </div>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>
        </div>
    );
}
