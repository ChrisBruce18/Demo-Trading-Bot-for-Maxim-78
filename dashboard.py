# dashboard.py
import matplotlib.pyplot as plt
import numpy as np

plt.ion()
fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
ax1.set_title("Price & Signals")
ax2.set_title("Cumulative P/L")
ax3.set_title("AI Commentary")

commentary_history = []

def update_dashboard(df, commentary=None):
    global commentary_history
    ax1.clear()
    ax2.clear()
    ax3.clear()

    ax1.plot(df['timestamp'], df['close'], label='Close Price', 
color='blue')
    ax1.plot(df['timestamp'], df['close'].rolling(5).mean(), label='MA 
Short', color='green')
    ax1.plot(df['timestamp'], df['close'].rolling(20).mean(), label='MA 
Long', color='red')

    buys = df[df['signal']=='buy']
    sells = df[df['signal']=='sell']
    if not buys.empty:
        ax1.scatter(buys['timestamp'], buys['close'], marker='^', 
color='green', label='Buy Signal')
    if not sells.empty:
        ax1.scatter(sells['timestamp'], sells['close'], marker='v', 
color='red', label='Sell Signal')
    ax1.legend(loc='upper left')

    df_plot = df.copy()
    df_plot['pl'] = np.where(df_plot['signal']=='buy', 1, 
np.where(df_plot['signal']=='sell', -1, 0))
    df_plot['cum_pl'] = df_plot['pl'].cumsum()
    ax2.plot(df_plot['timestamp'], df_plot['cum_pl'], color='purple', 
label='Cumulative P/L')
    ax2.legend(loc='upper left')

    if commentary:
        commentary_history.append(commentary)
        if len(commentary_history) > 6:
            commentary_history = commentary_history[-6:]
    ax3.text(0.01, 0.99, "\n".join(reversed(commentary_history)), 
fontsize=10, va='top', ha='left', wrap=True)
    ax3.axis('off')

    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

