package com.sign2speak.app

/**
 * GlossBuffer — accumule les glosses ASL et déclenche un flush après
 * [FLUSH_TIMEOUT_MS] ms de silence (aucun nouveau gloss).
 *
 * Thread-safety : cette classe n'est pas thread-safe par elle-même ;
 * elle est utilisée exclusivement depuis le thread d'analyse caméra
 * (single-threaded executor) dans MainViewModel.
 */
class GlossBuffer(private val flushTimeoutMs: Long = 3_000L) {

    private val glosses   = mutableListOf<String>()
    private var lastAddMs = 0L   // epoch ms du dernier push

    /**
     * Ajoute un gloss si différent du dernier enregistré.
     * Réinitialise le timer de silence.
     */
    fun push(gloss: String) {
        if (gloss.isBlank()) return
        if (glosses.isEmpty() || glosses.last() != gloss) {
            glosses.add(gloss)
        }
        lastAddMs = System.currentTimeMillis()
    }

    /**
     * À appeler à chaque frame (ou régulièrement).
     * Retourne la liste de glosses accumulées si le timeout est expiré,
     * et réinitialise le buffer. Retourne null si pas encore de flush.
     */
    fun tick(): List<String>? {
        if (glosses.isEmpty()) return null
        if (lastAddMs == 0L) return null

        val elapsed = System.currentTimeMillis() - lastAddMs
        if (elapsed >= flushTimeoutMs) {
            val snapshot = glosses.toList()
            glosses.clear()
            lastAddMs = 0L
            return snapshot
        }
        return null
    }

    /**
     * Retourne une copie de la liste courante (pour affichage, sans flush).
     */
    fun currentGlosses(): List<String> = glosses.toList()

    /**
     * Vide le buffer sans flush.
     */
    fun clear() {
        glosses.clear()
        lastAddMs = 0L
    }

    /** Vrai si au moins un gloss est en attente. */
    val isEmpty: Boolean get() = glosses.isEmpty()
}
