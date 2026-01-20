"""
Cache statique des films en France (janvier 2025)
Contient les films déjà en salles ET les films à venir
À utiliser quand l'API TMDb n'est pas accessible
"""

from datetime import datetime

# Date de référence (aujourd'hui)
DATE_REFERENCE = datetime.now()

# ==============================================
# FILMS DÉJÀ EN SALLES (sortie avant 16/01/2025)
# ==============================================

FILMS_EN_SALLES = [
    {
        'tmdb_id': 1241982,
        'titre': 'Vaiana 2',
        'titre_original': 'Moana 2',
        'poster_url': 'https://image.tmdb.org/t/p/w500/yh64qw9mgXBvlaWDi7Q9tpUBAvH.jpg',
        'synopsis': "Après avoir reçu un appel inattendu de ses ancêtres wayfinders, Moana doit voyager vers les mers lointaines d'Océanie et dans des eaux dangereuses et perdues pour une aventure différente de tout ce qu'elle a jamais affronté.",
        'note': 7.0,
        'nb_votes': 1542,
        'annee': 2024,
        'date_sortie': '2024-11-27',
        'realisateur': 'David G. Derrick Jr.',
        'acteurs': ["Auli'i Cravalho", 'Dwayne Johnson', 'Temuera Morrison', 'Nicole Scherzinger', 'Rachel House'],
        'genres': ['Animation', 'Aventure', 'Familial', 'Comédie'],
        'duree': 100,
        'langue_originale': 'en',
        'popularite': 2156.877,
        'statut': 'en_salles'
    },
    {
        'tmdb_id': 1002338,
        'titre': 'Le Comte de Monte-Cristo',
        'titre_original': 'Le Comte de Monte-Cristo',
        'poster_url': 'https://image.tmdb.org/t/p/w500/zw4kV7npGtaqvUxvJE9IdqdFsNc.jpg',
        'synopsis': "Edmond Dantès, un marin sur le point d'épouser l'amour de sa vie, Mercédès, est dénoncé par les trois hommes jaloux de son succès. Il est emprisonné dans le sinistre château d'If pendant quatorze ans. Après une évasion audacieuse, il découvre un trésor inestimable et prépare sa vengeance.",
        'note': 7.9,
        'nb_votes': 456,
        'annee': 2024,
        'date_sortie': '2024-06-28',
        'realisateur': 'Alexandre de La Patellière',
        'acteurs': ['Pierre Niney', 'Anaïs Demoustier', 'Bastien Bouillon', 'Laurent Lafitte', 'Pierfrancesco Favino'],
        'genres': ['Aventure', 'Drame', 'Histoire', 'Romance', 'Thriller'],
        'duree': 178,
        'langue_originale': 'fr',
        'popularite': 1687.543,
        'statut': 'en_salles'
    },
    {
        'tmdb_id': 762509,
        'titre': 'Mufasa : Le Roi Lion',
        'titre_original': 'Mufasa: The Lion King',
        'poster_url': 'https://image.tmdb.org/t/p/w500/lurEK87kukWNaHd0zYnsi3yzJrs.jpg',
        'synopsis': "Simba et Nala présentent leur petite-fille Kiara à Timon et Pumbaa qui lui racontent l'histoire de Mufasa. À travers des flashbacks, on découvre l'enfance de Mufasa, un lionceau orphelin perdu et seul, jusqu'à sa rencontre avec Taka, un lion de sang royal.",
        'note': 7.3,
        'nb_votes': 834,
        'annee': 2024,
        'date_sortie': '2024-12-18',
        'realisateur': 'Barry Jenkins',
        'acteurs': ['Aaron Pierre', 'Kelvin Harrison Jr.', 'Seth Rogen', 'Billy Eichner', 'John Kani'],
        'genres': ['Aventure', 'Familial', 'Drame', 'Animation'],
        'duree': 118,
        'langue_originale': 'en',
        'popularite': 1654.432,
        'statut': 'en_salles'
    },
    {
        'tmdb_id': 1064213,
        'titre': 'Sonic 3, le film',
        'titre_original': 'Sonic the Hedgehog 3',
        'poster_url': 'https://image.tmdb.org/t/p/w500/bXi6IQiQDHD00JFio5ZSZOeRSBh.jpg',
        'synopsis': "Sonic, Knuckles et Tails se retrouvent face à un nouvel adversaire de taille : Shadow, un mystérieux méchant doté de pouvoirs inédits. Leurs capacités mises à rude épreuve à tous points de vue, l'Équipe Sonic doit s'allier à un improbable associé pour arrêter Shadow et protéger la planète.",
        'note': 7.8,
        'nb_votes': 623,
        'annee': 2024,
        'date_sortie': '2024-12-19',
        'realisateur': 'Jeff Fowler',
        'acteurs': ['Ben Schwartz', 'Jim Carrey', 'Keanu Reeves', 'Idris Elba', "Colleen O'Shaughnessey"],
        'genres': ['Action', 'Aventure', 'Comédie', 'Familial', 'Science-Fiction'],
        'duree': 110,
        'langue_originale': 'en',
        'popularite': 1543.876,
        'statut': 'en_salles'
    },
    {
        'tmdb_id': 402431,
        'titre': 'Wicked',
        'titre_original': 'Wicked',
        'poster_url': 'https://image.tmdb.org/t/p/w500/xDGbZ0JJ3mYaGKy4Nzd9Kph6M9L.jpg',
        'synopsis': "Après avoir rencontré une jeune fille du nom d'Elphaba à l'université Shiz, la vie de la populaire et ambitieuse Glinda prend une tournure inattendue. Leur amitié improbable va les mener à travers les dangers du Pays d'Oz.",
        'note': 7.6,
        'nb_votes': 891,
        'annee': 2024,
        'date_sortie': '2024-11-20',
        'realisateur': 'Jon M. Chu',
        'acteurs': ['Cynthia Erivo', 'Ariana Grande', 'Jonathan Bailey', 'Michelle Yeoh', 'Jeff Goldblum'],
        'genres': ['Drame', 'Fantastique', 'Romance'],
        'duree': 160,
        'langue_originale': 'en',
        'popularite': 1432.654,
        'statut': 'en_salles'
    },
    {
        'tmdb_id': 558449,
        'titre': 'Gladiator 2',
        'titre_original': 'Gladiator II',
        'poster_url': 'https://image.tmdb.org/t/p/w500/2cxhvwyEwRlysAmRH4iodkvo0z5.jpg',
        'synopsis': "Des années après avoir été témoin de la mort héroïque du vénéré empereur Maximus aux mains de son oncle corrompu, Lucius est contraint d'entrer dans le Colisée après que sa maison ait été conquise par les empereurs tyranniques qui règnent maintenant sur Rome d'une main de fer.",
        'note': 7.1,
        'nb_votes': 1234,
        'annee': 2024,
        'date_sortie': '2024-11-13',
        'realisateur': 'Ridley Scott',
        'acteurs': ['Paul Mescal', 'Pedro Pascal', 'Denzel Washington', 'Connie Nielsen', 'Joseph Quinn'],
        'genres': ['Action', 'Aventure', 'Drame'],
        'duree': 148,
        'langue_originale': 'en',
        'popularite': 1398.765,
        'statut': 'en_salles'
    },
    {
        'tmdb_id': 957119,
        'titre': 'Nosferatu',
        'titre_original': 'Nosferatu',
        'poster_url': 'https://image.tmdb.org/t/p/w500/4WycdBWLJkwxJ3mNRfpMMqmpKR6.jpg',
        'synopsis': "Une histoire gothique d'obsession entre une jeune femme hantée et le terrifiant vampire qui en est épris, provoquant une horreur et une tragédie indicibles dans son sillage.",
        'note': 7.4,
        'nb_votes': 567,
        'annee': 2024,
        'date_sortie': '2024-12-25',
        'realisateur': 'Robert Eggers',
        'acteurs': ['Bill Skarsgård', 'Nicholas Hoult', 'Lily-Rose Depp', 'Aaron Taylor-Johnson', 'Emma Corrin'],
        'genres': ['Horreur', 'Fantastique'],
        'duree': 132,
        'langue_originale': 'en',
        'popularite': 1198.654,
        'statut': 'en_salles'
    },
    {
        'tmdb_id': 1184918,
        'titre': 'Le Robot sauvage',
        'titre_original': 'The Wild Robot',
        'poster_url': 'https://image.tmdb.org/t/p/w500/wTnV3PCVW5O92JMrFvvrRcV39RU.jpg',
        'synopsis': "Après avoir fait naufrage sur une île déserte, un robot nommé Roz doit apprendre à s'adapter à son nouvel environnement. En construisant des relations avec les animaux de l'île et en devenant le parent adoptif d'un oison orphelin, Roz découvre ce que signifie vraiment être en vie.",
        'note': 8.5,
        'nb_votes': 1876,
        'annee': 2024,
        'date_sortie': '2024-09-12',
        'realisateur': 'Chris Sanders',
        'acteurs': ["Lupita Nyong'o", 'Pedro Pascal', 'Kit Connor', 'Bill Nighy', 'Stephanie Hsu'],
        'genres': ['Animation', 'Science-Fiction', 'Familial'],
        'duree': 102,
        'langue_originale': 'en',
        'popularite': 876.543,
        'statut': 'en_salles'
    }
]

# ==============================================
# FILMS À VENIR (sortie après 16/01/2025)
# ==============================================

FILMS_A_VENIR = [
    {
        'tmdb_id': 845781,
        'titre': 'Red One : Mission secrète',
        'titre_original': 'Red One',
        'poster_url': 'https://image.tmdb.org/t/p/w500/cdqLnri3NEGcmfnqwk2TSIYtddg.jpg',
        'synopsis': "Après l'enlèvement du Père Noël - nom de code : RED ONE - le chef de la sécurité du Pôle Nord doit faire équipe avec le chasseur de primes le plus infâme du monde pour sauver Noël.",
        'note': 6.9,
        'nb_votes': 432,
        'annee': 2025,
        'date_sortie': '2025-01-22',
        'realisateur': 'Jake Kasdan',
        'acteurs': ['Dwayne Johnson', 'Chris Evans', 'Lucy Liu', 'J.K. Simmons', 'Bonnie Hunt'],
        'genres': ['Action', 'Comédie', 'Fantastique'],
        'duree': 123,
        'langue_originale': 'en',
        'popularite': 1876.234,
        'statut': 'bientot'
    },
    {
        'tmdb_id': 1184920,
        'titre': 'Sept Hommes à abattre',
        'titre_original': 'Seven Sinners',
        'poster_url': 'https://image.tmdb.org/t/p/w500/qVdrYN8qu7xUtsdEFeGiIVIJQuI.jpg',
        'synopsis': "Dans l'Amérique des années 1870, un groupe de sept hommes aux passés troubles se retrouve pour accomplir une mission périlleuse contre un baron du crime impitoyable.",
        'note': 7.2,
        'nb_votes': 234,
        'annee': 2025,
        'date_sortie': '2025-01-29',
        'realisateur': 'Peter Craig',
        'acteurs': ['Russell Crowe', 'Liam Hemsworth', 'Luke Hemsworth', 'Daniel MacPherson', 'Brooke Satchwell'],
        'genres': ['Action', 'Western', 'Thriller'],
        'duree': 135,
        'langue_originale': 'en',
        'popularite': 1543.876,
        'statut': 'bientot'
    },
    {
        'tmdb_id': 823219,
        'titre': 'Flow, le chat qui n\'avait plus peur de l\'eau',
        'titre_original': 'Flow',
        'poster_url': 'https://image.tmdb.org/t/p/w500/dzBtMocZuJbjLOXvrl4zGYigDzh.jpg',
        'synopsis': "Un chat se réveille dans un univers submergé où toute vie humaine semble avoir disparu. Il doit trouver refuge sur un bateau avec un groupe d'autres animaux et naviguer à travers ce monde étrange et dangereux.",
        'note': 8.7,
        'nb_votes': 234,
        'annee': 2025,
        'date_sortie': '2025-02-05',
        'realisateur': 'Gints Zilbalodis',
        'acteurs': [],  # Film d'animation sans acteurs vocaux principaux
        'genres': ['Animation', 'Aventure', 'Familial'],
        'duree': 85,
        'langue_originale': 'lv',
        'popularite': 1234.567,
        'statut': 'bientot'
    },
    {
        'tmdb_id': 558450,
        'titre': 'Captain America: Brave New World',
        'titre_original': 'Captain America: Brave New World',
        'poster_url': 'https://image.tmdb.org/t/p/w500/5qJPvlxsBBgvZVjrMN6kH6OGmqe.jpg',
        'synopsis': "Sam Wilson, qui a officiellement endossé le rôle de Captain America, se retrouve au milieu d'un incident international. Il doit découvrir la raison d'un complot mondial néfaste avant que le véritable cerveau ne fasse basculer le monde dans le chaos.",
        'note': 7.5,
        'nb_votes': 876,
        'annee': 2025,
        'date_sortie': '2025-02-12',
        'realisateur': 'Julius Onah',
        'acteurs': ['Anthony Mackie', 'Harrison Ford', 'Danny Ramirez', 'Carl Lumbly', 'Tim Blake Nelson'],
        'genres': ['Action', 'Aventure', 'Science-Fiction'],
        'duree': 128,
        'langue_originale': 'en',
        'popularite': 2345.123,
        'statut': 'bientot'
    },
    {
        'tmdb_id': 939243,
        'titre': 'Mickey 17',
        'titre_original': 'Mickey 17',
        'poster_url': 'https://image.tmdb.org/t/p/w500/7GdAzQUl6c8CvKzVC3aG94M5hFc.jpg',
        'synopsis': "Mickey Barnes a trouvé le travail parfait : il est un 'jetable', un employé sur une mission de colonisation humaine envoyé faire les travaux les plus dangereux. Après plusieurs morts, une nouvelle itération de Mickey prend sa place.",
        'note': 7.8,
        'nb_votes': 543,
        'annee': 2025,
        'date_sortie': '2025-03-07',
        'realisateur': 'Bong Joon-ho',
        'acteurs': ['Robert Pattinson', 'Naomi Ackie', 'Steven Yeun', 'Toni Collette', 'Mark Ruffalo'],
        'genres': ['Science-Fiction', 'Thriller', 'Comédie'],
        'duree': 142,
        'langue_originale': 'en',
        'popularite': 1987.654,
        'statut': 'bientot'
    }
]

# Liste combinée (pour compatibilité avec l'ancien code)
FILMS_AFFICHE_CACHE = FILMS_EN_SALLES + FILMS_A_VENIR


def separer_films_par_statut(films_list):
    """
    Sépare les films entre ceux déjà en salles et ceux à venir
    basé sur la date de sortie vs date de référence
    
    Args:
        films_list: Liste de films avec date_sortie
        
    Returns:
        tuple: (films_en_salles, films_a_venir)
    """
    en_salles = []
    a_venir = []
    
    for film in films_list:
        try:
            date_sortie_str = film.get('date_sortie', '')
            if date_sortie_str:
                date_sortie = datetime.strptime(date_sortie_str, '%Y-%m-%d')
                
                if date_sortie <= DATE_REFERENCE:
                    en_salles.append(film)
                else:
                    a_venir.append(film)
            else:
                # Si pas de date, considérer comme en salles
                en_salles.append(film)
        except:
            # En cas d'erreur, mettre en salles
            en_salles.append(film)
    
    return en_salles, a_venir

